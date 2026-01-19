import torch
import os
import json
import argparse
from transformers import AutoModelForCausalLM

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
            param_group_moment_dict['experts'][param_name] = [exp_avg_norm, exp_avg_sq_norm]
            # param_name: transformer.h.{i}.mlp.experts.{gate_proj, c_fc, c_proj}.
            # param_state['exp_avg']: [n_exp, n_embd, 4*n_embd] = [128, 512, 2048].

        elif 'router' in param_name:
            param_group_moment_dict['routers'][param_name] = [exp_avg_norm, exp_avg_sq_norm]
        else:
            param_group_moment_dict['others'][param_name]  = [exp_avg_norm, exp_avg_sq_norm]

for key in sorted(param_group_moment_dict.keys()):
    print(f"{key} parameters: {len(param_group_moment_dict[key])}")
    if param_group_moment_dict[key]:
        exp_avg_norms    = torch.tensor([m[0] for m in param_group_moment_dict[key].values()])
        exp_avg_sq_norms = torch.tensor([m[1] for m in param_group_moment_dict[key].values()])
        exp_avg_norm_mean = exp_avg_norms.mean().item()
        exp_avg_norm_std  = exp_avg_norms.std().item()
        exp_avg_sq_norm_mean = exp_avg_sq_norms.mean().item()
        exp_avg_sq_norm_std  = exp_avg_sq_norms.std().item()
        print(f"{key} exp_avg_norm:    {exp_avg_norm_mean:.4f}/{exp_avg_norm_std:.5f}")
        if args.verbose:
            print(f"{key} exp_avg_sq_norm: {exp_avg_sq_norm_mean:.6f}/{exp_avg_sq_norm_std:.7f}")
