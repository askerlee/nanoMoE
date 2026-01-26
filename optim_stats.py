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
parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint directory to load from")
parser.add_argument("--diff", type=str, default=None, help="Path to a second compared checkpoint directory to load from")
parser.add_argument("--topk", type=int, default=32, help="Top-K value for MoE (not used in this script)")
parser.add_argument("--bottomk", type=int, default=96, help="Bottom-K value for MoE (not used in this script)")
parser.add_argument("--verbose", action='store_true', help="Whether to print detailed parameter states")
args = parser.parse_args()

state_dict = torch.load(args.ckpt + "/training_state.pt", weights_only=False, map_location='cpu')
optim_state = state_dict['optimizer_state_dict']

# Load model weight.
model = AutoModelForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True)
if args.diff:
    model2 = AutoModelForCausalLM.from_pretrained(args.diff, trust_remote_code=True)
    print(f"{args.diff} loaded from second checkpoint for comparison.")
else:
    model2 = None

def create_param_dicts(model):
    # lm_head weight is tied to wte weight, so include it in embedding params
    embedding_suffixes = ('wte.weight', 'wpe.weight', 'lm_head.weight')

    # Split into decay, nodecay and embedding groups (same logic as optimizer).
    # This is to reconstruct parameter ordering as done in optimizer construction
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    handled_params = set()
    decay_params = []
    nodecay_params = []
    embedding_params = []

    for name, param in param_dict.items():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in handled_params:
            continue
        handled_params.add(pid)

        is_bias = name.endswith('bias')
        if name.endswith(embedding_suffixes):
            embedding_params.append([name, param])
        elif param.dim() >= 2 and not is_bias:
            decay_params.append([name, param])
        else:
            nodecay_params.append([name, param])

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

    if not os.path.exists("param_id_to_name.json"):
        with open("param_id_to_name.json", "w") as f:
            json.dump(param_id_to_name, f, indent=2)
        print("param_id_to_name.json created successfully.")

    return param_dict, param_id_to_name

def diff_weight(param_dict, param_dict2, param_name, param_name_short, 
                top_k_indices, bottom_k_indices, verbose):
    w1 = param_dict[param_name]
    w2 = param_dict2[param_name]
    top_diff    = w1[top_k_indices] - w2[top_k_indices]
    bottom_diff = w1[bottom_k_indices] - w2[bottom_k_indices]
    top_diff_norm    = top_diff.norm().item()
    bottom_diff_norm = bottom_diff.norm().item()
    print(f"{param_name_short} top    diff norm: {smart_format(top_diff_norm, 3)}")
    print(f"{param_name_short} bottom diff norm: {smart_format(bottom_diff_norm, 3)}")
    if verbose:
        dims = list(range(len(top_diff.shape)))[1:]
        top_diff_norms    = top_diff.norm(dim=dims)  # [K1]
        bottom_diff_norms = bottom_diff.norm(dim=dims)  # [K2]
        try:
            print(f"{param_name_short} top    diff norms:")
            print(", ".join([smart_format(v, precision=3) for v in top_diff_norms]))
            print(f"{param_name_short} bottom diff norms:")
            print(", ".join([smart_format(v, precision=3) for v in bottom_diff_norms]))
        except:
            breakpoint()

param_dict, param_id_to_name = create_param_dicts(model)
if model2:
    # model2's param_id_to_name should be the same as model. So no need to recreate.
    param_dict2 = {pn: p for pn, p in model2.named_parameters() if p.requires_grad}

# Get optimizer state
state        = optim_state['state']
param_groups = optim_state['param_groups']

param_group_stat_dict = { 'experts': {}, 'routers': {}, 'others': {} }

for param_id, param_state in state.items():
    param_name = param_id_to_name.get(str(param_id), f"unknown_{param_id}")
    if 'exp_avg' in param_state and 'exp_avg_sq' in param_state:
        # exp_avg_norm:    the norm of the first moment estimate
        # exp_avg_sq_norm: the norm of the second moment estimate
        exp_avg_norm = param_state['exp_avg'].norm().item()
        exp_avg_sq_norm = param_state['exp_avg_sq'].norm().item()

        if 'experts' in param_name:
            param_group_stat_dict['experts'][param_name] = {'exp_avg_norm': exp_avg_norm, 'exp_avg_sq_norm': exp_avg_sq_norm}
            # param_name: transformer.h.{i}.mlp.experts.{gate_proj, c_fc, c_proj}.
            # param_state['exp_avg']: [n_exp, n_embd, 4*n_embd] = [128, 512, 2048].
            experts_grad_norms_by_expert = param_state['exp_avg'].norm(dim=(1,2))  # [n_exp]
            # Find the top-32 and bottom-96 norms
            K1, K2 = args.topk, args.bottomk
            topk_experts_grad_norms, topk_indices       = torch.topk(experts_grad_norms_by_expert, K1)
            bottomk_experts_grad_norms, bottomk_indices = torch.topk(-experts_grad_norms_by_expert, K2)
            bottomk_experts_grad_norms      = -bottomk_experts_grad_norms
            topk_experts_grad_norm_mean     = topk_experts_grad_norms.mean().item()
            bottomk_experts_grad_norm_mean  = bottomk_experts_grad_norms.mean().item()
            param_group_stat_dict['experts'][param_name]['topk_experts_grad_norm_mean']          = topk_experts_grad_norm_mean
            param_group_stat_dict['experts'][param_name]['bottomk_experts_grad_norm_mean']       = bottomk_experts_grad_norm_mean
            param_group_stat_dict['experts'][param_name]['topk_bottomk_experts_grad_norm_ratio'] = topk_experts_grad_norm_mean / (bottomk_experts_grad_norm_mean + 1e-13)
            param_group_stat_dict['experts'][param_name]['experts_grad_norm_std']                = experts_grad_norms_by_expert.std().item()
            param_name_short = param_name.replace("transformer.h.", "").replace("mlp.experts.", "")
            grad_norm_ratio = param_group_stat_dict['experts'][param_name]['topk_bottomk_experts_grad_norm_ratio']
            grad_norm_std   = param_group_stat_dict['experts'][param_name]['experts_grad_norm_std']
            print(f"{param_name_short} top-{K1}/bottom-{K2} experts grad norm: {smart_format(topk_experts_grad_norm_mean)}/{smart_format(bottomk_experts_grad_norm_mean)} ratio: {smart_format(grad_norm_ratio, 4)}, std: {smart_format(grad_norm_std)}")

            if args.verbose:
                print(f"{param_name_short} Top-{K1}    experts indices: {topk_indices.tolist()}")
                print(f"{param_name_short} Bottom-{K2} experts indices: {bottomk_indices.tolist()}")

            if param_name.endswith('gate_proj'):
                router_gate_name = param_name.replace('experts.gate_proj', 'router.w_g.weight')
                router_gate_short_name = router_gate_name.replace("transformer.h.", "").replace("mlp.", "").replace(".weight", "")
                router_gate = param_dict[router_gate_name]
                # router_gate: [n_exp, n_emb]
                top_gate_norms = router_gate.norm(dim=1)[topk_indices]      # [K1]
                bottom_gate_norms = router_gate.norm(dim=1)[bottomk_indices]# [K2]
                top_gate_norm_mean = top_gate_norms.mean().item()
                bottom_gate_norm_mean = bottom_gate_norms.mean().item()
                print(f"{router_gate_short_name} top-{K1}    experts router gate norm mean: {smart_format(top_gate_norm_mean)}")
                print(f"{router_gate_short_name} bottom-{K2} experts router gate norm mean: {smart_format(bottom_gate_norm_mean)}")
                if args.verbose:
                    print(f"{router_gate_short_name} top-{K1}    experts router gate norms:")
                    print(", ".join([smart_format(v, precision=3) for v in top_gate_norms.tolist()]))
                    print(f"{router_gate_short_name} bottom-{K2} experts router gate norms:")
                    print(", ".join([smart_format(v, precision=3) for v in bottom_gate_norms.tolist()]))

                if model2:
                    # Diff router gate
                    diff_weight(param_dict, param_dict2, router_gate_name, router_gate_short_name,
                                topk_indices, bottomk_indices, args.verbose)
                    # Diff expert gate proj
                    diff_weight(param_dict, param_dict2, param_name, param_name_short,
                                topk_indices, bottomk_indices, args.verbose)
                    # Diff expert c_fc
                    expert_c_fc_name = param_name.replace('gate_proj', 'c_fc')
                    expert_c_fc_short_name = param_name_short.replace('gate_proj', 'c_fc')
                    diff_weight(param_dict, param_dict2, expert_c_fc_name, expert_c_fc_short_name,
                                topk_indices, bottomk_indices, args.verbose)
                    
            w = param_dict[param_name]
            # All such expert parameters should be 3D tensors like [n_exp, n_embd, 4*n_embd]
            if w.dim() != 3:
                breakpoint()
            w_norms = w.norm(dim=(1,2))  # [n_exp]
            w_norm_mean = w_norms.mean().item()
            param_group_stat_dict['experts'][param_name]['weight_norm_mean'] = w_norm_mean
            param_group_stat_dict['experts'][param_name]['weight_norm_std']  = w_norms.std().item()
            if args.verbose:
                print(f"{param_name_short} weight norm mean: {smart_format(w_norm_mean)}, std: {smart_format(w_norms.std().item())}")          

        elif 'router' in param_name:
            param_group_stat_dict['routers'][param_name] = {'exp_avg_norm': exp_avg_norm, 'exp_avg_sq_norm': exp_avg_sq_norm}
        else:
            param_group_stat_dict['others'][param_name]  = {'exp_avg_norm': exp_avg_norm, 'exp_avg_sq_norm': exp_avg_sq_norm}

for key in sorted(param_group_stat_dict.keys()):
    print(f"{key} parameters: {len(param_group_stat_dict[key])}")
    if param_group_stat_dict[key]:
        first_stat = next(iter(param_group_stat_dict[key].values()))
        for stat_name in first_stat.keys():
            if stat_name in ['exp_avg_norm', 'weight_norm_mean', 'weight_norm_std', 'topk_bottomk_experts_grad_norm_ratio']:
                stats      = torch.tensor([m[stat_name]    for m in param_group_stat_dict[key].values()])
                stats_mean = stats.mean().item()
                stats_std  = stats.std().item()
                print(f"{key} {stat_name}:    {stats_mean:.4f}/{stats_std:.5f}")
            if args.verbose:
                if stat_name in ['exp_avg_sq_norm']:
                    extra_stats = torch.tensor([m[stat_name] for m in param_group_stat_dict[key].values()])
                    extra_stats_mean = extra_stats.mean().item()
                    extra_stats_std  = extra_stats.std().item()
                    print(f"{key} {stat_name}: {extra_stats_mean:.6f}/{extra_stats_std:.7f}")
