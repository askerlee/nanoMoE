import os
import time
import torch
from modeling_nanomoe_gpt import GPTConfig, GPT, MOELayer, Qwen3MLPExperts

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
ckpt_prefix = "nanomoe"
seed = 1337

# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'nano-moe'
wandb_run_name = 'gpt2-124M-owt' + str(time.time())

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
use_router_ortho_loss = False
use_experts_ortho_loss = False
use_gate_output_loss = False
use_noisy_top_k = False
aux_loss_weight = 0.001
router_z_loss_weight = 0.01
router_ortho_loss_weight = 0.01
router_ortho_neg_corr_weight = 1  # weight for negative correlations in router-ortho loss
# experts_ortho_loss is very small due to squared cosine similarities.
# So its weight is set higher to have a meaningful effect.
experts_ortho_loss_weight = 0.01  
gate_output_loss_weight = 0.0001
train_capacity = 1.25
eval_capacity = 3.0
min_capacity = 4
stride = 2
moe_start_layer = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = False
use_qwen3_moe_mlp = False

# adamw optimizer
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0

# epoch-based training
num_epochs = 1.0  # total number of epochs to train (can be fractional)
evals_per_epoch = 10  # number of evaluations per epoch
warmup_tokens = 20_000_000  # absolute number of tokens for warmup (20M)
decay_frac = 0.1     # fraction of total steps used for final decay

# learning rate schedule
decay_lr = True  # whether to use the warmup/stable/decay schedule

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# NOTE: Always override 'resume_from' in the command line or config file to load a checkpoint for eigenvalue analysis.
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

assert resume_from is not None, "Please specify a checkpoint to load using the 'resume_from' variable."

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, n_exp=n_exp, moe_top_k=moe_top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_router_ortho_loss=use_router_ortho_loss,
                  use_experts_ortho_loss=use_experts_ortho_loss,
                  use_gate_output_loss=use_gate_output_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, 
                  router_ortho_loss_weight=router_ortho_loss_weight,
                  router_ortho_neg_corr_weight=router_ortho_neg_corr_weight,
                  experts_ortho_loss_weight=experts_ortho_loss_weight,
                  gate_output_loss_weight=gate_output_loss_weight,
                  train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, 
                  stride=stride, moe_start_layer=moe_start_layer,
                  use_switch_tfm_init=use_switch_tfm_init, switch_tfm_init_scale=switch_tfm_init_scale,
                  router_use_full_prec=router_use_full_prec,
                  use_qwen3_moe_mlp=use_qwen3_moe_mlp) # start with model_args from command line
print('\n\n')
print(model_args)
print('\n\n')
# Load model
model = GPT.from_pretrained(resume_from, trust_remote_code=True)
model.to(device)

row_sim_means = []
row_sim_stds = []

# Remove trailing slash if any
resume_from = resume_from.rstrip('/').rstrip('\\')
ckpt_filename = os.path.basename(resume_from)
print(f"Model {ckpt_filename}:")

for layer, block in enumerate(model.transformer.h):
    if not isinstance(block.mlp, MOELayer):
        #print(f"Block {layer} is not a MOELayer, skipping...")
        continue
    moe_layer = block.mlp
    assert isinstance(moe_layer.experts, Qwen3MLPExperts), "Expected Qwen3MLPExperts"
    experts = moe_layer.experts
    # gate_projs: [n_exp, n_embd, intermediate_size]
    gate_projs = experts.gate_proj
    row_sims = []

    for i in range(experts.n_exp):
        gate_proj = gate_projs[i]  # [n_embd, intermediate_size]
        G = gate_proj
        eps = 1e-12
        G = G / (G.norm(dim=1, keepdim=True) + eps)   # row-normalize
        gram = G @ G.T
        offdiag = gram - torch.eye(gram.size(0), device=gram.device)
        row_sim = offdiag.square().mean()
        row_sims.append(row_sim.item())
    
    row_sims_tensor = torch.tensor(row_sims)
    row_sim_mean = row_sims_tensor.mean().item()
    row_sim_std = row_sims_tensor.std().item()
    row_sim_means.append(row_sim_mean)
    row_sim_stds.append(row_sim_std)
    # Compute the stats of top 5 highest row similarities for this layer
    top5_row_sims, _ = torch.topk(row_sims_tensor, 5)
    top5_row_sim_mean = top5_row_sims.mean().item()
    top5_row_sim_std = top5_row_sims.std().item()
    print(f"Layer {layer}: Row sim mean = {row_sim_mean:.6f}, std = {row_sim_std:.6f}. Top 5 mean = {top5_row_sim_mean:.6f}, std = {top5_row_sim_std:.6f}")

overall_mean = torch.tensor(row_sim_means).mean().item()
overall_std  = torch.tensor(row_sim_stds).mean().item()
print(f"\nOverall Row sim mean = {overall_mean:.6f}, std = {overall_std:.6f}")
