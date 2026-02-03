"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
import time
import math
import json
from contextlib import nullcontext
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2TokenizerFast

from modeling_nanomoe_gpt import GPTConfig, GPT
from manager import MANAGER
from data.tinystories.dataloader import ChunkDataset, BoundaryChunkDataset
import numpy as np
import random

def seed_worker(worker_seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

# Create deterministic sampler for reproducible shuffling with DDP support
def get_epoch_sampler(dataset, epoch, seed, ddp_world_size=1, ddp_rank=0):
    """Create a sampler with deterministic shuffling based on epoch, sharded for DDP"""
    if ddp_world_size > 1:
        # Use DistributedSampler for proper DDP data sharding
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        sampler.set_epoch(epoch)
        return sampler
    else:
        # Single-GPU: use SubsetRandomSampler with deterministic shuffling
        g = torch.Generator()
        g.manual_seed(seed + epoch)  # Different seed per epoch
        indices = torch.randperm(len(dataset), generator=g).tolist()
        return torch.utils.data.SubsetRandomSampler(indices)


class SkipBatchSampler(torch.utils.data.Sampler):
    """Wrap a BatchSampler and skip the first N batches without loading data."""

    def __init__(self, batch_sampler, skip_batches=0):
        self.batch_sampler = batch_sampler
        self.skip_batches = max(0, int(skip_batches))

    def __iter__(self):
        for i, batch in enumerate(self.batch_sampler):
            if i < self.skip_batches:
                continue
            yield batch

    def __len__(self):
        return max(0, len(self.batch_sampler) - self.skip_batches)


def build_train_loader(dataset, sampler, batch_size, num_workers, device_type, seed, seed_offset, skip_batches=0):
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
    if skip_batches:
        batch_sampler = SkipBatchSampler(batch_sampler, skip_batches)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True if device_type == 'cuda' else False,
        worker_init_fn=lambda worker_id: seed_worker(seed + seed_offset + worker_id)
    )

def collect_grad_stats(model, losses, moe_start_layer, n_layer):
    router_grad_norms = []
    router_grad_self_alignments = []
    router_weight_exp_alignments = []
    exp_gate_grad_norms = []
    expert_utilities = losses.get('expert_utilities', None)
    selected_scores = losses.get('selected_scores', None)

    for i in range(moe_start_layer, n_layer):
        layer = model.transformer.h[i]
        if hasattr(layer.mlp, 'experts'):
            # [n_exp, hidden_size]
            router_gate_grad = layer.mlp.router.w_g.weight.grad
            router_grad_norm = router_gate_grad.norm(dim=1)
            router_grad_norms.append(router_grad_norm)
            losses[f'router_grad_norm_{i}'] = router_grad_norm.mean().item()
            exp_gate_grad = layer.mlp.experts.gate_proj.grad
            exp_gate_grad_norm = exp_gate_grad.norm(dim=(1,2))
            exp_gate_grad_norms.append(exp_gate_grad_norm)
            losses[f'exp_gate_grad_norm_{i}'] = exp_gate_grad_norm.mean().item()

            # Compute router grad - router weight alignment
            # Compute router expert - gate weight alignment
            with torch.no_grad():
                router_weight = layer.mlp.router.w_g.weight  # [n_exp, hidden_size]
                exp_gate_mean_weight = layer.mlp.experts.gate_proj.mean(dim=2)  # [n_exp, hidden_size]
                # Compute the cosine similarity between router weights and router weight grads.
                # With SGD: Δw = -lr * ∇w. Since w·Δw = -lr*(w·∇w),
                # -(w·∇w) is positive when the update has a component along w (tends to increase ||w||),
                # and negative when it moves against w (tends to decrease ||w||). 
                rg_rw_alignment = -(router_gate_grad * router_weight).sum(dim=1) / (
                    router_weight.norm(dim=1) * router_gate_grad.norm(dim=1) + 1e-10
                )  # [n_exp]
                router_grad_self_alignments.append(rg_rw_alignment)
                mean_rg_rw_alignment = rg_rw_alignment.mean().item()
                losses[f'router_grad_self_alignment_{i}'] = mean_rg_rw_alignment

                # No negative sign here since these are weights, not gradients.
                rw_ew_alignment = (exp_gate_mean_weight * router_weight).sum(dim=1) / \
                        (router_weight.norm(dim=1) * (exp_gate_mean_weight.norm(dim=1) + 1e-10)) # [n_exp]
                router_weight_exp_alignments.append(rw_ew_alignment)
                mean_rw_ew_alignment = rw_ew_alignment.mean().item()
                losses[f'router_weight_exp_alignment_{i}'] = mean_rw_ew_alignment

                if expert_utilities is not None:
                    # expert_utilities: Tensor of shape (num_moe_layers, n_exp)
                    exp_utilities = expert_utilities[i - moe_start_layer]  # [n_exp]
                    half_experts = exp_utilities.shape[0] // 2
                    top_indices    = torch.topk(exp_utilities, k=half_experts, largest=True).indices
                    bottom_indices = torch.topk(exp_utilities, k=half_experts, largest=False).indices

                    top_rg_rw_alignment    = rg_rw_alignment[top_indices].mean().item()
                    bottom_rg_rw_alignment = rg_rw_alignment[bottom_indices].mean().item()
                    losses[f'router_grad_self_alignment_top_{i}']    = top_rg_rw_alignment
                    losses[f'router_grad_self_alignment_bottom_{i}'] = bottom_rg_rw_alignment

                    top_rw_ew_alignment    = rw_ew_alignment[top_indices].mean().item()
                    bottom_rw_ew_alignment = rw_ew_alignment[bottom_indices].mean().item()
                    losses[f'router_weight_exp_alignment_top_{i}']    = top_rw_ew_alignment
                    losses[f'router_weight_exp_alignment_bottom_{i}'] = bottom_rw_ew_alignment

                    top_router_grad_norm    = router_grad_norm[top_indices].mean().item()
                    bottom_router_grad_norm = router_grad_norm[bottom_indices].mean().item()
                    losses[f'router_grad_norm_top_{i}']    = top_router_grad_norm
                    losses[f'router_grad_norm_bottom_{i}'] = bottom_router_grad_norm

                    if selected_scores is not None:
                        # selected_scores: Tensor of shape (num_moe_layers, n_exp)
                        layer_selected_scores = selected_scores[i - moe_start_layer]  # [n_exp]
                        top_selected_scores    = layer_selected_scores[top_indices].mean().item()
                        bottom_selected_scores = layer_selected_scores[bottom_indices].mean().item()
                        losses[f'selected_scores_top_{i}']    = top_selected_scores
                        losses[f'selected_scores_bottom_{i}'] = bottom_selected_scores

    router_grad_norms = torch.stack(router_grad_norms, dim=0) if router_grad_norms else None
    losses['router_grad_norms'] = router_grad_norms
    router_grad_self_alignments = torch.stack(router_grad_self_alignments, dim=0) if router_grad_self_alignments else None
    losses['router_grad_self_alignments'] = router_grad_self_alignments
    router_weight_exp_alignments = torch.stack(router_weight_exp_alignments, dim=0) if router_weight_exp_alignments else None
    losses['router_weight_exp_alignments'] = router_weight_exp_alignments
    exp_gate_grad_norms = torch.stack(exp_gate_grad_norms, dim=0) if exp_gate_grad_norms else None
    losses['exp_gate_grad_norms'] = exp_gate_grad_norms

# expert_utilities:           Tensor of shape (num_eval_batches, num_moe_layers, n_exp).
# router_ortho_losses_by_exp: Tensor of shape (num_eval_batches, num_moe_layers, n_exp).
def write_expert_util_stats(expert_utilities: torch.Tensor, 
                            selected_scores: torch.Tensor,
                            router_ortho_losses_by_exp: torch.Tensor, 
                            router_grad_self_alignments: torch.Tensor,
                            router_weight_exp_alignments: torch.Tensor,
                            router_grad_norms: torch.Tensor,
                            exp_gate_grad_norms: torch.Tensor,
                            tag: str, filepath: str) -> None:
    """Persist expert utilization stats to disk under a given tag."""
    record = {
        "tag": tag,
    }    

    if expert_utilities is not None:
        expert_utilities_str_list = [ [f"{u:08.5f}" for u in layer] for layer in expert_utilities.detach().cpu().tolist() ]
        record["expert_utilities"] = expert_utilities_str_list
    else:
        expert_utilities_str_list = None

    if selected_scores is not None:
        selected_scores_str_list = [ [f"{s:08.5f}" for s in layer] for layer in selected_scores.detach().cpu().tolist() ]
        record["selected_scores"] = selected_scores_str_list
    else:
        selected_scores_str_list = None

    if router_ortho_losses_by_exp is not None:
        router_ortho_losses_by_exp_str_list = [ [f"{l:08.5f}" for l in layer] for layer in router_ortho_losses_by_exp.detach().cpu().tolist() ]
        record["router_ortho_losses_by_exp"] = router_ortho_losses_by_exp_str_list
    else:
        router_ortho_losses_by_exp_str_list = None

    if router_grad_norms is not None:
        router_grad_norms_str_list = [ [f"{g:08.5f}" for g in layer] for layer in router_grad_norms.detach().cpu().tolist() ]
        record["router_grad_norms"] = router_grad_norms_str_list
    else:
        router_grad_norms_str_list = None

    if router_grad_self_alignments is not None:
        router_grad_self_alignments_str_list = [ [f"{g:08.5f}" for g in layer] for layer in router_grad_self_alignments.detach().cpu().tolist() ]
        record["router_grad_self_alignments"] = router_grad_self_alignments_str_list
    else:
        router_grad_self_alignments_str_list = None

    if router_weight_exp_alignments is not None:
        router_weight_exp_alignments_str_list = [ [f"{g:08.5f}" for g in layer] for layer in router_weight_exp_alignments.detach().cpu().tolist() ]
        record["router_weight_exp_alignments"] = router_weight_exp_alignments_str_list
        
    if exp_gate_grad_norms is not None:
        # exp_gate_grad_norms are close to 0, so scale by 1000 for better readability.
        exp_gate_grad_norms_str_list = [ [f"{g*1000:08.5f}" for g in layer] for layer in exp_gate_grad_norms.detach().cpu().tolist() ]
        record["exp_gate_grad_norms"] = exp_gate_grad_norms_str_list
    else:
        exp_gate_grad_norms_str_list = None
        
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")


# estimate validation loss using many batches
@torch.no_grad()
def estimate_loss(model, val_loader, max_eval_batches):
    model.eval()

    num_eval_batches = min(max_eval_batches, len(val_loader)) if max_eval_batches > 0 else len(val_loader)
    num_moe_layers = model.config.n_layer - model.config.moe_start_layer
    moe_top_k = model.config.moe_top_k

    # Scalar losses are always present; pre-allocate them
    # Tensor losses start as None and are lazily allocated if present in model output
    val_losses = {
        'ntp_loss':             torch.zeros(num_eval_batches, device=device),
        'aux_loss':             torch.zeros(num_eval_batches, device=device),
        'router_z_loss':        torch.zeros(num_eval_batches, device=device),
        'router_ortho_loss':    torch.zeros(num_eval_batches, device=device),
        'experts_ortho_loss':   torch.zeros(num_eval_batches, device=device),
        'gate_output_loss':     torch.zeros(num_eval_batches, device=device),
        'projs_diversity_loss': torch.zeros(num_eval_batches, device=device),
        'router_ortho_losses_by_exp':   None,
        'drop_rate_per_ks':             None,
        'expert_utilities':             None,
        'selected_scores':              None,
    }

    for k, (X, Y) in enumerate(val_loader):
        if k >= num_eval_batches:
            break
        
        if k % log_interval == 0:
            # Only collect drop_rate_per_ks across every log_interval iters to reduce overhead.
            MANAGER.collect_load_balancing_stats = True
        else:
            MANAGER.collect_load_balancing_stats = False

        # Move to device
        if device_type == 'cuda':
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
            
        with ctx:
            _, loss, losses = model(input_ids=X, labels=Y, return_dict=False)
        
        # Accumulate losses. Most auxiliary losses are zero during eval since
        # they are computed only when self.training is True in modeling_nanomoe_gpt.py.
        for key in val_losses:
            if key == 'drop_rate_per_ks':
                if losses.get(key) is not None:
                    if val_losses[key] is None:
                        val_losses[key] = torch.zeros((num_eval_batches, moe_top_k), device=device)
                    val_losses[key][k] = losses[key]
            elif key in ('expert_utilities', 'router_ortho_losses_by_exp', 'selected_scores'):
                if losses.get(key) is not None:
                    if val_losses[key] is None:
                        val_losses[key] = torch.zeros((num_eval_batches, num_moe_layers, model.config.n_exp), device=device)
                    val_losses[key][k] = losses[key]
            else:
                val_losses[key][k] = losses[key]
    
    model.train()
    MANAGER.collect_load_balancing_stats = False
    # Mean over eval batches; scalar losses become scalars, tensor losses keep remaining dims
    return {key: (val_losses[key].mean(dim=0) if val_losses[key] is not None else None) for key in val_losses}

# learning rate scheduler (warmup -> stable -> decay to zero)
def get_lr(learning_rate: float, it: int) -> float:
    """Compute learning rate at iteration it."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / float(warmup_iters + 1)
    if it < decay_start:
        return learning_rate
    if it >= total_iters:
        return 0.0
    decay_ratio = (it - decay_start) / float(max(1, decay_iters))
    return learning_rate * (1 - math.sqrt(decay_ratio))

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) with epoch-based training
# I/O
out_dir = 'out'
log_interval = 25
eval_only = False # if True, script exits right after the first eval
save_ckpt_every_n_evals = 50 # if -1, never save checkpoints
# if True, always save a checkpoint after each eval, no matter whether the val loss is optimal.
save_ckpt_regardless_loss = True 
# Whether to save optimizer and scaler state along with model.
# Useful on slurm clusters where jobs have a short time limit and need to be resumed often.
save_training_state = True  
skip_batches_on_resume = True  # if True, skip batches when resuming to avoid repeating training on the same data.
ckpt_prefix = "nanomoe"
seed = 1337

# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'nano-moe'
wandb_run_name = 'gpt2-124M-owt' + str(time.time())
# Whether to save expert utilization stats from beginning to this iteration.
# Set to -1 to disable.
log_expert_util_stats_until_training_iter = -1
log_expert_util_stats_during_eval = False
log_grad_stats = False # Use command line to override this if needed.

# data
# To set datasets in the command line, use e.g. --datasets="['fineweb_edu-30b']".
# Note we need to use double quotes outside and single quotes inside to make it a valid string for the shell.
datasets = ['fineweb_edu-50B'] #, 'openwebtext'] #'tinystories', 'openwebtext', 'fineweb_edu-30B', 'fineweb_edu-50B'
gradient_accumulation_steps = 2 # used to simulate larger batch sizes
batch_size = 12     # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024   # Training tokens per sample

# model
n_layer = 12
n_head = 12
n_embd = 768
bias = False # do we use bias inside LayerNorm and Linear layers?

# moe
n_exp = 1 # if n_exp = 1 we just use regular MLP layers
moe_top_k = 2
use_aux_loss = False
use_router_z_loss = False
use_logits_demeaned_z_loss = True
penalize_pos_mean_logits = True
use_router_ortho_loss = False
use_experts_ortho_loss = False
use_gate_output_loss = False
use_noisy_top_k = False
aux_loss_weight = 0.001
router_z_loss_weight = 0.00001
router_ortho_loss_weight = 0.01
router_ortho_neg_corr_weight = 1  # weight for negative correlations in router-ortho loss
# experts_ortho_loss is very small due to squared cosine similarities.
# So its weight is set higher to have a meaningful effect.
experts_ortho_loss_weight = 0.01  
gate_output_loss_weight = 0.0001
projs_diversity_loss_weight = 0.01
train_capacity = 1.25
eval_capacity = 3.0
# min_capacity: minimum number of tokens per expert. 
# train_capacity and eval_capacity are scale factors, not absolute numbers.
min_capacity = 4 
stride = 2
moe_start_layer = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = False
use_qwen3_moe_mlp = False

use_muon = True

if use_muon:
    learning_rate = 1e-2
    weight_decay = 0
else:
    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    weight_decay = 1e-1

lr_scale = 1.0  # scale learning rate by this factor, convenient for continual training with lower lr.
beta1 = 0.9     # NOTE: nanochat uses a default 0.8.
beta2 = 0.95
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0

# epoch-based training
num_epochs = 1.0  # total number of epochs to train (can be fractional)
evals_per_epoch = 500  # number of evaluations per epoch
# eval_every_n_iters will be set based on evals_per_epoch if -1
# If eval_every_n_iters is provided, will set evals_per_epoch based on it.
eval_every_n_iters = -1 
# max number of training batches, -1 means use full epoch.
# Set if we want to limit training time for debugging.
max_training_batches = -1 
max_eval_batches = -1 # number of batches to use for eval, -1 means use full val set
warmup_tokens = 500_000_000  # absolute number of tokens for warmup (500M)
decay_frac = 0.1     # fraction of total steps used for final decay
load_optimizer_state = True

# learning rate schedule
decay_lr = True  # whether to use the warmup/stable/decay schedule

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
resume_from = None  # Override to resume from a checkpoint directory.

# profiling
use_profiler = False # enable PyTorch profiler
profiler_schedule_wait = 2 # number of steps to wait before profiling
profiler_schedule_warmup = 2 # number of warmup steps
profiler_schedule_active = 6 # number of active profiling steps
profiler_schedule_repeat = 1 # number of times to repeat the schedule
profiler_output_dir = './profiler_results' # directory to save profiler results
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# Remove non-existent variables that were removed during epoch-based conversion
config_keys = [k for k in config_keys if k not in ['max_iters', 'lr_decay_iters', 'eval_interval']]
# Put everything in argv[1:] into globals(). If an argument is a config file, exec it,
# otherwise if it's a --key=value argument, override the corresponding key in globals().
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print(config)
# -----------------------------------------------------------------------------

current_folder = os.path.dirname(os.path.abspath(__file__))
log_expert_util_stats_file = os.path.join(current_folder, f'{ckpt_prefix}-expert-utils.jsonl')

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
seed_worker(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_flash_sdp(True)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if isinstance(datasets, str):
    datasets = [datasets]

train_datasets = []
val_datasets = []

for dataset in datasets:
    dataset_ = dataset
    # data loading
    if "-" in dataset:
        # Only split at the first '-' to allow dataset names like "fineweb_edu-50B-skip50B"
        dataset, dataset_size = dataset.split("-", 1)
        train_filename = f"train-{dataset_size}.bin"
        val_filename   = f"val-{dataset_size}.bin"
    else:
        train_filename = "train.bin"
        val_filename   = "val.bin"
        
    data_dir = os.path.join('data', dataset)
    train_bin_path = os.path.join(data_dir, train_filename)
    val_bin_path = os.path.join(data_dir, val_filename)

    if "math" not in dataset:
        train_dataset = ChunkDataset(train_bin_path, block_size)
        val_dataset = ChunkDataset(val_bin_path, block_size)
    else:
        train_idx_path = train_bin_path.replace('.bin', '.idx')
        val_idx_path   = val_bin_path.replace('.bin', '.idx')
        train_dataset  = BoundaryChunkDataset(train_bin_path, train_idx_path, block_size)
        val_dataset    = BoundaryChunkDataset(val_bin_path, val_idx_path, block_size)

    print(f"Loaded dataset {dataset_} ({train_bin_path}, {val_bin_path}):")
    print(f"  train tokens: {len(train_dataset.data):,}")
    print(f"  val tokens: {len(val_dataset.data):,}")
    print()
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)

# Combine multiple datasets, then create DataLoaders
# ConcatDataset concatenates datasets sequentially, then shuffle=True mixes samples
# This allows batches to contain samples from different base datasets
num_workers = min(4, os.cpu_count() or 1)
combined_train_dataset = torch.utils.data.ConcatDataset(train_datasets)

# Don't use shuffle=True, use sampler instead for reproducible shuffling.
# batch_size = 64
# block_size = 1024
# Each batch contains 64*1024 = 64K tokens.
train_sampler = get_epoch_sampler(combined_train_dataset, epoch=0, seed=seed, 
                                   ddp_world_size=ddp_world_size, ddp_rank=ddp_rank if ddp else 0)
train_loader = build_train_loader(
    combined_train_dataset,
    train_sampler,
    batch_size,
    num_workers,
    device_type,
    seed,
    seed_offset
)

combined_val_dataset = torch.utils.data.ConcatDataset(val_datasets)
val_loader = torch.utils.data.DataLoader(
    combined_val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # Keep validation single-threaded for deterministic results
    pin_memory=True if device_type == 'cuda' else False,
    drop_last=True
)

print(f"Using {len(combined_train_dataset)} training tokens from datasets: {datasets}")
print(f"Using {len(combined_val_dataset)} validation tokens from datasets: {datasets}")
print(f"Train batches per epoch: {len(train_loader)}")
print(f"Validation batches per eval: {len(val_loader)}")

# Calculate epoch parameters
iters_per_epoch = len(train_loader)
total_iters = int(num_epochs * iters_per_epoch)
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
warmup_iters = int(warmup_tokens / tokens_per_iter)  # Convert tokens to iterations
decay_iters = int(decay_frac * total_iters)
decay_start = total_iters - decay_iters

if eval_every_n_iters == -1:
    eval_every_n_iters = max(1, iters_per_epoch // evals_per_epoch)
else:
    evals_per_epoch = iters_per_epoch // eval_every_n_iters

tokens_per_epoch = tokens_per_iter * iters_per_epoch

print(f"Epoch configuration:")
print(f"  Iterations per epoch: {iters_per_epoch}")
print(f"  Num epochs: {num_epochs}")
print(f"  Total iterations: {total_iters}")
print(f"  Warmup iters: {warmup_iters}")
print(f"  Decay iters: {decay_iters}")
print(f"  Evaluations per epoch: {evals_per_epoch}")
print(f"  Evaluate every {eval_every_n_iters} iterations")
print(f"  Tokens per iteration: {tokens_per_iter:,}")
print(f"  Tokens per epoch: {tokens_per_epoch:,}")



# training state
best_val_loss = 1e9
global_iter = 0
# persist_global_iter: persistent global iteration counter across training sessions.
persist_global_iter = 0
eval_count = 0
start_epoch = 0
start_batch_idx = 0

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, n_exp=n_exp, moe_top_k=moe_top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_logits_demeaned_z_loss=use_logits_demeaned_z_loss,
                  penalize_pos_mean_logits=penalize_pos_mean_logits,
                  use_router_ortho_loss=use_router_ortho_loss,
                  use_experts_ortho_loss=use_experts_ortho_loss,
                  use_gate_output_loss=use_gate_output_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, 
                  router_ortho_loss_weight=router_ortho_loss_weight,
                  router_ortho_neg_corr_weight=router_ortho_neg_corr_weight,
                  experts_ortho_loss_weight=experts_ortho_loss_weight,
                  gate_output_loss_weight=gate_output_loss_weight,
                  projs_diversity_loss_weight=projs_diversity_loss_weight,
                  train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, 
                  stride=stride, moe_start_layer=moe_start_layer,
                  use_switch_tfm_init=use_switch_tfm_init, switch_tfm_init_scale=switch_tfm_init_scale,
                  router_use_full_prec=router_use_full_prec,
                  use_qwen3_moe_mlp=use_qwen3_moe_mlp) # start with model_args from command line
print('\n\n')
print(model_args)
print('\n\n')

# Check if resuming from checkpoint or training from scratch
training_state = None
if resume_from:
    print(f"Resuming training from {resume_from}")
    
    # Load model from checkpoint
    model = GPT.from_pretrained(resume_from, trust_remote_code=True)
    model.to(device)

    if os.path.exists(os.path.join(resume_from, 'training_state.pt')):
        # Load training state
        # PyTorch 2.6+ defaults weights_only=True; training_state contains non-tensor objects.
        # Set weights_only=False when loading trusted checkpoints.
        training_state = torch.load(
            os.path.join(resume_from, 'training_state.pt'),
            map_location='cpu',
            weights_only=False,
        )
        global_iter = training_state['global_iter']
        persist_global_iter = training_state.get('persist_global_iter', global_iter)
        eval_count = training_state['eval_count']
        best_val_loss = training_state['best_val_loss']
        start_epoch = training_state.get('epoch', 0)
        start_batch_idx = training_state.get('batch_idx', 0)
        if skip_batches_on_resume and start_batch_idx > 0:
            print(f"Will skip {start_batch_idx} batches in epoch {start_epoch} to resume correctly.")
        else:
            print(f"Checkpoint indicates to skip {start_batch_idx} batches, but skip_batches_on_resume is set to False.")
            start_batch_idx = 0
            # NOTE: don't reset persist_global_iter here, instead allow persist_global_iter to 
            # track the total number of iterations across multiple training sessions.
            global_iter = 0
        
        # Restore RNG states
        rng_state = training_state['rng_state']
        if isinstance(rng_state, torch.Tensor):
            rng_state = rng_state.detach().to(device='cpu', dtype=torch.uint8)
        else:
            rng_state = torch.as_tensor(rng_state, dtype=torch.uint8, device='cpu')
        torch.set_rng_state(rng_state)
        np.random.set_state(training_state['numpy_rng_state'])
        random.setstate(training_state['python_rng_state'])
        if 'cuda_rng_state' in training_state and torch.cuda.is_available():
            cuda_rng_state = training_state['cuda_rng_state']
            if isinstance(cuda_rng_state, torch.Tensor):
                cuda_rng_state = cuda_rng_state.detach().to(device='cpu', dtype=torch.uint8)
            else:
                cuda_rng_state = torch.as_tensor(cuda_rng_state, dtype=torch.uint8, device='cpu')
            torch.cuda.set_rng_state(cuda_rng_state)
        print(f"Resumed from epoch {start_epoch}, batch {start_batch_idx}, iter {global_iter}")
    # Keep training_state for loading optimizer/scaler states later
else:
    # Train from scratch - determine the vocab size and create new model
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
base_lr = learning_rate * lr_scale

# optimizer(s) - model.setup_optimizers may return a single optimizer or a list
if use_muon:
    # The nanochat optimizer setup.
    optimizer_result = model.setup_optimizers(embedding_lr=base_lr, matrix_lr=base_lr * 0.1,
                                              weight_decay=weight_decay, adam_betas=(beta1, beta2))
else:
    # The old nanoMoE optimizer setup.
    optimizer_result = model.setup_optimizers2(base_lr, weight_decay, (beta1, beta2))

if isinstance(optimizer_result, (list, tuple)):
    optimizers = list(optimizer_result)
else:
    optimizers = [optimizer_result]

# Load optimizer and scaler state if resuming
if load_optimizer_state and training_state is not None:
    optimizer_state_dicts = training_state['optimizer_state_dict']
    if not isinstance(optimizer_state_dicts, list):
        optimizer_state_dicts = [optimizer_state_dicts]
    for optimizer, state_dict in zip(optimizers, optimizer_state_dicts):
        optimizer.load_state_dict(state_dict)        
        # Discard gradient accumulation buffers if any, to avoid OOM.
        optimizer.zero_grad(set_to_none=True)
    
    scaler.load_state_dict(training_state['scaler_state_dict'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("tokens_seen")
    wandb.define_metric("train/*", step_metric="tokens_seen")
    wandb.define_metric("val/*", step_metric="tokens_seen")

# training loop
t0 = time.time()
running_mfu = -1.0
running_tokens_per_sec = -1.0  # EMA of tokens processed per second

# Initialize profiler if enabled
profiler = None
if use_profiler and master_process:
    os.makedirs(profiler_output_dir, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device_type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    profiler = profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=profiler_schedule_wait,
            warmup=profiler_schedule_warmup,
            active=profiler_schedule_active,
            repeat=profiler_schedule_repeat
        ),
        on_trace_ready=lambda trace: (
            torch.profiler.tensorboard_trace_handler(profiler_output_dir)(trace),
            trace.export_chrome_trace(os.path.join(profiler_output_dir, f"trace_{int(time.time())}.json"))
        )[-1],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.start()

raw_model = model.module if ddp else model

stop_training = False
for epoch in range(start_epoch, math.ceil(num_epochs)):
    # If resuming mid-epoch, skip batches
    skip_batches = start_batch_idx if epoch == start_epoch else 0

    # Recreate sampler with epoch-specific seed for all epochs except the first (already created above)
    if epoch != 0:
        train_sampler = get_epoch_sampler(combined_train_dataset, epoch, seed,
                                         ddp_world_size=ddp_world_size, ddp_rank=ddp_rank if ddp else 0)
    else:
        # For epoch 0, just update the existing sampler if using DistributedSampler
        if isinstance(train_sampler, torch.utils.data.distributed.DistributedSampler):
            train_sampler.set_epoch(epoch)

    # Build loader; if resuming mid-epoch, skip in the sampler (no data loading overhead)
    train_loader = build_train_loader(
        combined_train_dataset,
        train_sampler,
        batch_size,
        num_workers,
        device_type,
        seed,
        seed_offset,
        skip_batches=skip_batches
    )

    if master_process:
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        if skip_batches > 0:
            print(f"Skipping {skip_batches} batches to resume from checkpoint")

    if master_process:
        pbar = tqdm(
            enumerate(train_loader, start=skip_batches),
            initial=skip_batches,
            total=iters_per_epoch,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )
    else:
        pbar = enumerate(train_loader, start=skip_batches)
        
    for batch_idx, (X, Y) in pbar:
        if (global_iter >= total_iters) or (max_training_batches > 0 and batch_idx >= max_training_batches):
            if max_training_batches > 0 and batch_idx >= max_training_batches:
                stop_training = True
            break
        grad_norm = None  # Initialize gradient norm for logging
        
        # Move to device
        if device_type == 'cuda':
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)

        # determine and set the learning rate for this iteration
        lr = get_lr(learning_rate * lr_scale, global_iter) if decay_lr else (learning_rate * lr_scale)
        for i, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                # The first optimizer is always adamw, 
                if i == 0:
                    param_group['lr'] = lr
                else:
                    param_group['lr'] = lr * optimizer.base_lr / optimizers[0].base_lr

        # evaluate the loss on train/val sets and write checkpoints
        if global_iter > 0 and global_iter % eval_every_n_iters == 0 and master_process:
            eval_count += 1
            val_losses = estimate_loss(model, val_loader, max_eval_batches=max_eval_batches)
            print(f"epoch {epoch + 1}, step {global_iter}: val loss {val_losses['ntp_loss']:.4f}")
            if wandb_log:
                log_data = {
                    "val/loss": val_losses['ntp_loss'],
                    "val/eval_count": eval_count,
                    "tokens_seen": persist_global_iter * batch_size * block_size,
                }
                drop_rates = val_losses['drop_rate_per_ks']
                if drop_rates is not None:
                    if len(drop_rates) >= 1:
                        log_data["val/drop_rate_0_step"] = drop_rates[0]
                    if len(drop_rates) >= 2:
                        log_data["val/drop_rate_1_step"] = drop_rates[1]
                wandb.log(log_data, step=persist_global_iter)
                if log_expert_util_stats_during_eval:
                    write_expert_util_stats(val_losses['expert_utilities'], 
                                            val_losses['selected_scores'],
                                            val_losses['router_ortho_losses_by_exp'],
                                            None, # No router_weight_exp_alignments during eval
                                            None, # No router_grad_norms during eval
                                            None, # No router_grad_self_alignments during eval
                                            None, # No exp_gate_grad_norms during eval
                                            f"val-{persist_global_iter:06d}", 
                                            log_expert_util_stats_file)

            if save_ckpt_every_n_evals != -1 and (val_losses['ntp_loss'] < best_val_loss or save_ckpt_regardless_loss) and (eval_count % save_ckpt_every_n_evals == 0):
                best_val_loss = val_losses['ntp_loss']
                
                # Save model using HuggingFace format
                ckpt_dir = os.path.join(out_dir, f'{ckpt_prefix}-{eval_count}')
                print(f"saving checkpoint to {ckpt_dir}")
                raw_model.save_pretrained(ckpt_dir)
                
                # Save tokenizer files
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                tokenizer.model_max_length = block_size
                tokenizer.save_pretrained(ckpt_dir)
                
                # Copy necessary files for trust_remote_code loading
                import shutil
                for filename in ['configuration_nanomoe_gpt.py', 'modeling_nanomoe_gpt.py', 'manager.py']:
                    src = os.path.join(os.path.dirname(__file__), filename)
                    dst = os.path.join(ckpt_dir, filename)
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                print(f"copied model files for trust_remote_code loading")
                
                if save_training_state:
                    # Add to training state
                    # Save list of optimizer state dicts if multiple optimizers, else single state dict
                    optimizer_state_dict = [opt.state_dict() for opt in optimizers] if len(optimizers) > 1 else optimizers[0].state_dict()
                    training_state = {
                        'global_iter': global_iter,
                        'persist_global_iter': persist_global_iter,
                        'eval_count': eval_count,
                        'best_val_loss': best_val_loss,
                        'optimizer_state_dict': optimizer_state_dict,
                        'scaler_state_dict': scaler.state_dict(),
                        'config': config,
                        'model_args': model_args,
                        'epoch': epoch,
                        'batch_idx': batch_idx,  # track position within epoch
                        'rng_state': torch.get_rng_state(),  # PyTorch RNG state
                        'numpy_rng_state': np.random.get_state(),  # NumPy RNG state
                        'python_rng_state': random.getstate(),  # Python RNG state
                        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    }
                    torch.save(training_state, os.path.join(ckpt_dir, 'training_state.pt'))

        if eval_only and epoch == 0 and batch_idx == 0:
            # Run one evaluation then exit
            val_losses = estimate_loss(model, val_loader, max_eval_batches=max_eval_batches)
            print(f"eval_only mode: val loss {val_losses['ntp_loss']:.4f}")
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        
        # Handle DDP gradient sync - only sync when we're about to step the optimizer
        if ddp:
            model.require_backward_grad_sync = ((global_iter + 1) % gradient_accumulation_steps == 0)
        
        # Forward and backward pass for this batch
        MANAGER.collect_load_balancing_stats = (global_iter % log_interval == 0)
        raw_model.global_iter = global_iter  # for debugging purposes
        with record_function("forward_backward"):
            with ctx:
                with record_function("forward"):
                    logits, loss, losses = model(input_ids=X, labels=Y, return_dict=False)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    
            # backward pass, with gradient scaling if training in fp16
            with record_function("backward"):
                scaler.scale(loss).backward()

        if log_grad_stats and MANAGER.collect_load_balancing_stats:
            collect_grad_stats(raw_model, losses, moe_start_layer, n_layer)

        # Only step optimizer(s) every gradient_accumulation_steps iterations
        if (global_iter + 1) % gradient_accumulation_steps == 0:
            with record_function("optimizer_step"):
                # disable gradient clipping for now
                #if grad_clip != 0.0:
                #    scaler.unscale_(optimizer)
                #    # Store gradient norm tensor for later logging (avoid .item() sync here)
                #    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # step the optimizer(s) and scaler if training in fp16

                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if global_iter % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            grad_normf = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else None
            # compute tokens per second for this iteration (raw)
            tokens_ps = (batch_size * block_size) / dt if dt > 0 else 0.0

            # update running averages once the loop has warmed up a few iterations (parallels MFU logic)
            if global_iter >= 5:  # let the training loop settle a bit before smoothing
                running_tokens_per_sec = tokens_ps if running_tokens_per_sec == -1.0 else 0.9 * running_tokens_per_sec + 0.1 * tokens_ps
                mfu = raw_model.estimate_mfu(batch_size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            # Update tqdm progress bar with loss, tok/s, and MFU
            pbar.set_postfix({
                'loss': f'{lossf:.4f}',
                'tok/s': tqdm.format_sizeof(running_tokens_per_sec, divisor=1000),
                'mfu': f'{running_mfu*100:.2f}%'
            })
            
            if wandb_log:
                log_data = {
                    "train/loss_step": losses['ntp_loss'],
                    "train/grad_norm": grad_normf,
                    "train/aux_loss_step": losses['aux_loss'],
                    "train/router_z_loss_step": losses['router_z_loss'],
                    "train/router_ortho_loss_step": losses['router_ortho_loss'],
                    "train/experts_ortho_loss_step": losses['experts_ortho_loss'],
                    "train/gate_output_loss_step": losses['gate_output_loss'],
                    "train/projs_diversity_loss_step": losses['projs_diversity_loss'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "tok_per_sec": running_tokens_per_sec,
                    "time_ms": dt*1000,
                    "tokens_seen": persist_global_iter * batch_size * block_size,
                }
                drop_rates = losses['drop_rate_per_ks']
                if drop_rates is not None:
                    if len(drop_rates) >= 1:
                        log_data["inspect/drop_rate_0_step"] = drop_rates[0]
                    if len(drop_rates) >= 2:
                        log_data["inspect/drop_rate_1_step"] = drop_rates[1]
                for i in range(moe_start_layer, n_layer):
                    if f'router_grad_norm_top_{i}' in losses:
                        log_data.update({f"inspect/router_grad_norm_top_{i}": losses[f'router_grad_norm_top_{i}']})
                    if f'router_grad_norm_bottom_{i}' in losses:
                        log_data.update({f"inspect/router_grad_norm_bottom_{i}": losses[f'router_grad_norm_bottom_{i}']})
                    if f'router_grad_self_alignment_top_{i}' in losses:
                        log_data.update({f"inspect/router_grad_self_alignment_top_{i}": losses[f'router_grad_self_alignment_top_{i}']})
                    if f'router_grad_self_alignment_bottom_{i}' in losses:
                        log_data.update({f"inspect/router_grad_self_alignment_bottom_{i}": losses[f'router_grad_self_alignment_bottom_{i}']})
                    if f'router_weight_exp_alignment_top_{i}' in losses:
                        log_data.update({f"inspect/router_weight_exp_alignment_top_{i}": losses[f'router_weight_exp_alignment_top_{i}']})
                    if f'router_weight_exp_alignment_bottom_{i}' in losses:
                        log_data.update({f"inspect/router_weight_exp_alignment_bottom_{i}": losses[f'router_weight_exp_alignment_bottom_{i}']})
                    if f'selected_scores_top_{i}' in losses:
                        log_data.update({f"inspect/selected_scores_top_{i}": losses[f'selected_scores_top_{i}']})
                    if f'selected_scores_bottom_{i}' in losses:
                        log_data.update({f"inspect/selected_scores_bottom_{i}": losses[f'selected_scores_bottom_{i}']})
                        
                wandb.log(log_data, step=persist_global_iter)

            if log_expert_util_stats_until_training_iter > 0 and persist_global_iter <= log_expert_util_stats_until_training_iter:
                write_expert_util_stats(losses['expert_utilities'], 
                                        losses['selected_scores'],
                                        losses['router_ortho_losses_by_exp'],
                                        losses['router_grad_self_alignments'] if log_grad_stats else None,
                                        losses['router_weight_exp_alignments'] if log_grad_stats else None,
                                        losses['router_grad_norms'] if log_grad_stats else None,
                                        losses['exp_gate_grad_norms'] if log_grad_stats else None,
                                        f"train-{persist_global_iter:06d}", 
                                        log_expert_util_stats_file)

            MANAGER.collect_load_balancing_stats = False
        
        # Profiler step
        if profiler is not None:
            profiler.step()

        global_iter += 1
        persist_global_iter += 1

    if stop_training or global_iter >= total_iters:
        break

# Stop profiler if it was started
if profiler is not None:
    profiler.stop()
    print(f"Profiler results saved to {profiler_output_dir}")

if ddp:
    destroy_process_group()
