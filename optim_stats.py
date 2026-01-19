import torch
import os
import json
import argparse
from transformers import AutoModelForCausalLM

def smart_format(value, precision=5):
    """Format number in scientific notation if very small/large, otherwise normal."""
    if value == 0:
        return f"{0:.{precision}e}"
    abs_val = abs(value)
    # Use scientific notation if < 0.001 or > 10000
    if abs_val < 1e-3 or abs_val > 1e4:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

parser = argparse.ArgumentParser()
parser.add_argument("--resume_from", type=str, required=True, help="Path to the checkpoint directory to resume from")
parser.add_argument("--verbose", action='store_true', help="Whether to print detailed parameter states")
args = parser.parse_args()

state_dict = torch.load(args.resume_from + "/training_state.pt", weights_only=False, map_location='cpu')
optim_state = state_dict['optimizer_state_dict']

if not os.path.exists("param_id_to_name.json"):
    print("param_id_to_name.json not found in the checkpoint directory. Creating it now...")
    model = AutoModelForCausalLM.from_pretrained(args.resume_from, trust_remote_code=True)

    # Reconstruct parameter ordering as done in optimizer construction
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Split into decay and nodecay groups (same logic as optimizer)
    decay_params = [(n, p) for n, p in param_dict.items() if (p.dim() >= 2 and not n.endswith('bias'))]
    nodecay_params = [(n, p) for n, p in param_dict.items() if (p.dim() < 2 or n.endswith('bias'))]

    # Build param_id_to_name matching optimizer's parameter order
    # Optimizer groups are: [decay_params, nodecay_params]
    param_id_to_name = {}
    param_id = 0
    for name, param in decay_params:
        param_id_to_name[str(param_id)] = name
        param_id += 1
    for name, param in nodecay_params:
        param_id_to_name[str(param_id)] = name
        param_id += 1

    with open("param_id_to_name.json", "w") as f:
        json.dump(param_id_to_name, f, indent=2)
    print("param_id_to_name.json created successfully.")
else:
    with open("param_id_to_name.json", "r") as f:
        param_id_to_name = json.load(f)

# Get optimizer state
state        = optim_state['state']
param_groups = optim_state['param_groups']

param_group_moment_dict = { 'experts': {}, 'routers': {}, 'others': {} }

for param_id, param_state in state.items():
    param_name = param_id_to_name.get(str(param_id), f"unknown_{param_id}")
    if 'exp_avg' in param_state and 'exp_avg_sq' in param_state:
        # exp_avg_norm:    the norm of the first moment estimate
        # exp_avg_sq_norm: the norm of the second moment estimate
        exp_avg_norm = param_state['exp_avg'].norm().item()
        exp_avg_sq_norm = param_state['exp_avg_sq'].norm().item()

        if 'experts' in param_name:
            param_group_moment_dict['experts'][param_name] = {'exp_avg_norm': exp_avg_norm, 'exp_avg_sq_norm': exp_avg_sq_norm}
            # param_name: transformer.h.{i}.mlp.experts.{gate_proj, c_fc, c_proj}.
            # param_state['exp_avg']: [n_exp, n_embd, 4*n_embd] = [128, 512, 2048].
            exp_avg_norms_by_expert = param_state['exp_avg'].norm(dim=(1,2))  # [n_exp]
            # Find the top-32 and bottom-96 norms
            K1, K2 = 32, 96
            topk_norms = torch.topk(exp_avg_norms_by_expert, K1).values
            bottomk_norms = - torch.topk(-exp_avg_norms_by_expert, K2).values
            topk_norm_mean    = topk_norms.mean().item()
            bottomk_norm_mean = bottomk_norms.mean().item()
            param_group_moment_dict['experts'][param_name]['topk_exp_avg_norm_mean']    = topk_norm_mean
            param_group_moment_dict['experts'][param_name]['bottomk_exp_avg_norm_mean'] = bottomk_norm_mean
            param_group_moment_dict['experts'][param_name]['topk_bottomk_ratio'] = topk_norm_mean / (bottomk_norm_mean + 1e-13)
            param_group_moment_dict['experts'][param_name]['exp_avg_norm_std'] = exp_avg_norms_by_expert.std().item()
            param_name_short = param_name.replace("transformer.h.", "").replace("mlp.experts.", "")
            ratio = param_group_moment_dict['experts'][param_name]['topk_bottomk_ratio']
            std = param_group_moment_dict['experts'][param_name]['exp_avg_norm_std']
            print(f"{param_name_short} top-{K1}/bottom-{K2} exp_avg_norm: {smart_format(topk_norm_mean)}/{smart_format(bottomk_norm_mean)} ratio: {smart_format(ratio, 4)}, std: {smart_format(std)}")

        elif 'router' in param_name:
            param_group_moment_dict['routers'][param_name] = {'exp_avg_norm': exp_avg_norm, 'exp_avg_sq_norm': exp_avg_sq_norm}
        else:
            param_group_moment_dict['others'][param_name]  = {'exp_avg_norm': exp_avg_norm, 'exp_avg_sq_norm': exp_avg_sq_norm}

for key in sorted(param_group_moment_dict.keys()):
    print(f"{key} parameters: {len(param_group_moment_dict[key])}")
    if param_group_moment_dict[key]:
        exp_avg_norms    = torch.tensor([m['exp_avg_norm']    for m in param_group_moment_dict[key].values()])
        exp_avg_sq_norms = torch.tensor([m['exp_avg_sq_norm'] for m in param_group_moment_dict[key].values()])
        exp_avg_norm_mean = exp_avg_norms.mean().item()
        exp_avg_norm_std  = exp_avg_norms.std().item()
        exp_avg_sq_norm_mean = exp_avg_sq_norms.mean().item()
        exp_avg_sq_norm_std  = exp_avg_sq_norms.std().item()
        print(f"{key} exp_avg_norm:    {exp_avg_norm_mean:.4f}/{exp_avg_norm_std:.5f}")
        if args.verbose:
            print(f"{key} exp_avg_sq_norm: {exp_avg_sq_norm_mean:.6f}/{exp_avg_sq_norm_std:.7f}")
