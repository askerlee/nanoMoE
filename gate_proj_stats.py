import os
import time
import torch
import argparse
from transformers import AutoModelForCausalLM
from modeling_nanomoe_gpt import MOELayer, Qwen3MLPExperts

parser = argparse.ArgumentParser()
parser.add_argument("--resume_from", type=str, required=True, help="Path to the checkpoint directory to resume from")
parser.add_argument("--calc_scheme", type=str, default="batch", choices=["loop", "batch", "rand_estimate"],
                    help="Scheme to calculate row similarity: 'loop' for per-expert loop, 'batch' for batched Gram, "
                         "'rand_estimate' for randomized estimation")
parser.add_argument("--verbose", action='store_true', help="Whether to print detailed parameter states")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.resume_from, trust_remote_code=True)
row_sim_means = []
row_sim_stds = []

# Remove trailing slash if any
resume_from = args.resume_from.rstrip('/').rstrip('\\')
ckpt_filename = os.path.basename(resume_from)
print(f"Model {ckpt_filename}:")

num_rand_probes = 16

for layer, block in enumerate(model.transformer.h):
    if not isinstance(block.mlp, MOELayer):
        #print(f"Block {layer} is not a MOELayer, skipping...")
        continue
    moe_layer = block.mlp
    assert isinstance(moe_layer.experts, Qwen3MLPExperts), "Expected Qwen3MLPExperts"
    experts = moe_layer.experts
    # G: [n_exp, n_embd, intermediate_size]
    G = experts.gate_proj
    # Row-normalize: normalize each row vector over intermediate_size
    G = G / (G.norm(dim=2, keepdim=True) + 1e-12)
    E, D, F = G.size()  # n_exp, n_embd, intermediate_size

    if args.calc_scheme == 'rand_estimate':
        est_frob2 = torch.zeros(E, device=G.device, dtype=G.dtype)
        for _ in range(num_rand_probes):
            z = (torch.randint(0, 2, (E, D, 1), device=G.device) * 2 - 1).to(G.dtype)  # [E,D,1]
            gt_z = torch.bmm(G.transpose(1, 2), z)   # [E,F,1]
            Az   = torch.bmm(G, gt_z) - z            # [E,D,1]
            est_frob2 += Az.squeeze(-1).square().sum(dim=1)  # [E]

        est_frob2 /= num_rand_probes

        # Match offdiag.square().mean() (mean over D*D entries)
        row_sim_per_expert = est_frob2 / (D * D - D)

    elif args.calc_scheme == 'batch':
        # Batched Gram: [n_exp, n_embd, n_embd]
        # This computes cosine similarity between all pairs of row vectors of 
        # each expert's gate projection matrix.
        gram = torch.bmm(G, G.transpose(1, 2))
        # Zero out diagonal without materializing eye per expert
        gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
        # Mean squared off-diagonal similarity per expert
        # Off-diagonal count = n_embd * n_embd - n_embd
        offdiag_sq_sum = gram.square().sum(dim=(1, 2))      # [n_exp]
        row_sim_per_expert = offdiag_sq_sum / (D * D - D)   # [n_exp]
    else:
        # Loop over experts
        row_sim_per_expert = []

        for i in range(experts.n_exp):
            Gi = G[i]  # [n_embd, intermediate_size]
            eps = 1e-12
            gram = Gi @ Gi.T
            offdiag = gram - torch.eye(gram.size(0), device=gram.device)
            row_sim = offdiag.square().sum() / (D * D - D)
            row_sim_per_expert.append(row_sim.item())
        
        row_sim_per_expert = torch.tensor(row_sim_per_expert, device=G.device)

    row_sim_mean = row_sim_per_expert.mean().item()
    row_sim_std = row_sim_per_expert.std().item()
    row_sim_means.append(row_sim_mean)
    row_sim_stds.append(row_sim_std)
    # Compute the stats of top 5 highest row similarities for this layer
    top5_row_sims, _ = torch.topk(row_sim_per_expert, 5)
    top5_row_sim_mean = top5_row_sims.mean().item()
    top5_row_sim_std = top5_row_sims.std().item()
    print(f"Layer {layer}: Row sim mean = {row_sim_mean:.6f}, std = {row_sim_std:.6f}. Top 5 mean = {top5_row_sim_mean:.6f}, std = {top5_row_sim_std:.6f}")

overall_mean = torch.tensor(row_sim_means).mean().item()
overall_std  = torch.tensor(row_sim_stds).mean().item()
print(f"\nOverall Row sim mean = {overall_mean:.6f}, std = {overall_std:.6f}")
