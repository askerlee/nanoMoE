import time

# config for training GPT-2 (124M) baseline model (one expert) on two RTX 3090 GPUs
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_nano_moe.py

seed = 1337

wandb_log = True
init_from = 'scratch'
wandb_project = 'nano-moe'
wandb_run_name ='gpt2-124M-moe-owt ' + time.strftime('%Y-%m-%d %H:%M:%S')

# model/moe settings
n_exp = 128
moe_top_k = 2
# aux loss is most useful for load balancing.
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
# router z loss helps reduce overfitting. 
# But the effect diminishes quickly as the weight increases.
# So the weight is small.
router_z_loss_weight = 0.0002
use_router_ortho_loss = True
# experts_ortho_loss is slow to compute and has negative effect on router_ortho_loss.
use_experts_ortho_loss = False 
# gate_output_loss always makes things slightly worse in our experiments. So turn it off by default.
use_gate_output_loss = True
# After changing to mean, the router orthogonality loss is tiny (<0.05), 
# so maybe we should increase its weight.
router_ortho_loss_weight = 0.01
router_ortho_neg_corr_weight = 1
# Experts orthogonality loss as of arXiv:2601.00457.
# experts_ortho_loss is very small due to squared cosine similarities.
# So its weight is set higher to have a meaningful effect.
experts_ortho_loss_weight = 0.01
gate_output_loss_weight = 0.0001
projs_diversity_loss_weight = 0.01

use_noisy_top_k = False
train_capacity = 1
eval_capacity = 3.0
use_switch_tfm_init = True
router_use_full_prec = True
use_qwen3_moe_mlp = True

# use smaller GPT model
n_layer = 8
# The first two layers are dense layers.
moe_start_layer = 2 
# Since layer 2, all mlps are MoEs.
stride = 1
n_head = 8
n_embd = 512

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 2

# epoch-based training
num_epochs = 1.0
evals_per_epoch = 500
warmup_frac = 0.01
decay_frac = 0.1

# logging and eval stuff
eval_iters = 200
log_interval = 25

# weight decay
weight_decay = 1e-1
