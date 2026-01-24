"""
Full definition of a NanoMoE GPT Language Model.
This file contains all the model components for HuggingFace compatibility.
"""

import math
import inspect
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from .manager import MANAGER
except ImportError:
    from manager import MANAGER
from transformers.activations import SiLUActivation
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from .configuration_nanomoe_gpt import GPTConfig
except ImportError:
    from configuration_nanomoe_gpt import GPTConfig

from functools import partial
from dataclasses import dataclass
import inspect
from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# Revised from RevGrad, by removing the grad negation.
class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_, debug=False):
        ctx.save_for_backward(alpha_, debug)
        output = input_
        if debug:
            print(f"input: {input_.abs().mean().detach().item()}")
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        # saved_tensors returns a tuple of tensors.
        alpha_, debug = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_output2 = grad_output * alpha_
            if debug:
                print(f"grad_output2: {grad_output2.abs().mean().detach().item()}")
        else:
            grad_output2 = None
        return grad_output2, None, None

class GradientScaler(nn.Module):
    def __init__(self, alpha=1., debug=False, *args, **kwargs):
        """
        A gradient scaling layer.
        This layer has no parameters, and simply scales the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)
        self._debug = torch.tensor(debug, requires_grad=False)

    def forward(self, input_):
        _debug = self._debug if hasattr(self, '_debug') else False
        return ScaleGrad.apply(input_, self._alpha.to(input_.device), _debug)

def gen_gradient_scaler(alpha, debug=False):
    if alpha == 1:
        return nn.Identity()
    if alpha > 0:
        return GradientScaler(alpha, debug=debug)
    else:
        assert alpha == 0
        # Don't use lambda function here, otherwise the object can't be pickled.
        return torch.detach

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.head_dim = config.n_embd // config.n_head
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # split into heads first …
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK norm
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.moe_top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss           = config.use_aux_loss
        self.use_router_z_loss      = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None

    def forward(self, x):
        """
        Computes routing information for tokens, including which experts to use,
        the weights for their outputs, and their position within the expert's batch.
        This implementation is memory-efficient and avoids quadratic scaling with batch size.
        """
        # The router can be sensitive to precision issues, so we can run it in full float32.
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, C = x.size()
            num_tokens = B * T
            x_flat = x.view(num_tokens, C)

            # 1. GET ROUTING LOGITS
            # ---------------------
            logits = self.w_g(x_flat)  # [B*T, n_exp]
            if self.training and self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x_flat))
                noise *= torch.randn_like(noise)
                logits += noise

            # 2. COMPUTE LOSSES (if training)
            # -------------------------------
            if self.training:
                # Router Z-loss prevents logits from growing too large
                if self.use_router_z_loss:
                    z_loss = self.compute_router_z_loss(logits.view(B, T, -1))
                    MANAGER.add_router_z_loss(z_loss)

                # Find top-k choices for each token
                top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B*T, k]
                
                # The auxiliary loss encourages load balancing across experts
                if self.use_aux_loss:
                    # To compute aux loss, we need the full probability distribution,
                    # not just for the top k. We can create this sparsely.
                    all_probs = torch.zeros_like(logits)
                    all_probs.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))
                    aux_loss = self.compute_aux_loss(all_probs.view(B, T, -1), top_k_indices.view(B, T, -1))
                    MANAGER.add_aux_loss(aux_loss)
            else:
                 # At inference, we just need the top-k
                 top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

            # 3. COMPUTE ROUTER PROBABILITIES
            # --------------------------------
            # We normalize the probabilities over the top-k experts
            router_probs = F.softmax(top_k_logits, dim=-1) # [B*T, k]

            # 4. DETERMINE TOKEN RANKS WITH CAPACITY LIMITING
            # -----------------------------------------------
            exp_capacity = self.get_capacity(num_tokens)
            
            # Create a one-hot mask of the chosen experts for each token. Shape: [B*T, k, n_exp]
            expert_mask_one_hot = F.one_hot(top_k_indices, num_classes=self.n_exp)

            # This is the critical step to ensure load balancing prioritizes top-1 experts.
            # We flatten the k dimension first, so cumsum processes all top-1 choices, then all top-2, etc.
            # This is the memory-efficient equivalent of the original logic.
            reshaped_mask = expert_mask_one_hot.permute(1, 0, 2).reshape(self.top_k * num_tokens, self.n_exp)
            cumulative_sum = torch.cumsum(reshaped_mask, dim=0)
            
            # Reshape back to the original layout
            position_in_expert = cumulative_sum.reshape(self.top_k, num_tokens, self.n_exp).permute(1, 0, 2)
            
            # The rank is the position, but we only care about the rank for the selected expert.
            # We multiply by the one-hot mask to zero out positions for non-selected experts.
            rank = (position_in_expert - 1) * expert_mask_one_hot
            
            # 5. GENERATE FINAL MASKS AND RANKS FOR THE MOE LAYER
            # ----------------------------------------------------
            # Create a mask to drop tokens that exceed the expert's capacity
            capacity_mask = rank < exp_capacity

            # The final expert mask includes both the expert choice and the capacity check.
            final_expert_mask = expert_mask_one_hot * capacity_mask # [B*T, k, n_exp]
            
            # Router probabilities are also masked. If a token is dropped, its probability is zero.
            # We check if the token was assigned to any expert in its k-th slot.
            probs_mask = (final_expert_mask.sum(dim=-1) > 0) # [B*T, k]
            router_probs_masked = router_probs * probs_mask

            # The final rank is collapsed to a single value per top-k choice
            final_rank = torch.sum(rank, dim=-1) # [B*T, k]

            # The MOELayer will use these tensors to efficiently dispatch and combine tokens.
            # Their memory usage all scale linearly with (B * T).
            return final_expert_mask, router_probs_masked, top_k_indices, final_rank 
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class ReLUSquared(nn.Module):
    """ReLU-squared activation: computes ReLU(x) then squares the result (x * x)."""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x, inplace=self.inplace)
        return x * x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Use ReLU-squared activation
        self.act = ReLUSquared()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        # Use ReLU-squared activation
        self.act = ReLUSquared()
    
    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.act(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        return x

# Borrowed Qwen3MoeMLP implementation from modeling_qwen3_moe.py.
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # up_proj -> c_fc, down_proj -> c_proj
        # to ensure minimal code changes when switching between Qwen3MoeMLP and regular MLP.
        self.c_fc = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        down_proj = self.c_proj(self.act_fn(self.gate_proj(x)) * self.c_fc(x))
        return down_proj

class Qwen3MLPExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_exp = config.n_exp
        self.hidden_size = config.n_embd
        self.intermediate_size = 4 * config.n_embd

        self.gate_proj = nn.Parameter(torch.empty(self.n_exp, self.hidden_size, self.intermediate_size))
        self.c_fc = nn.Parameter(torch.empty(self.n_exp, self.hidden_size, self.intermediate_size))
        self.c_proj = nn.Parameter(torch.empty(self.n_exp, self.intermediate_size, self.hidden_size))

        self.act_fn = SiLUActivation()
        self.fc_bias = None
        self.proj_bias = None
        self.gate_output_loss = 0
        self.use_gate_output_loss = config.use_gate_output_loss
        self.grad_scaler = gen_gradient_scaler(0.1)

    def forward(self, x):
        gate_out = torch.bmm(x, self.gate_proj)
        if self.use_gate_output_loss:
            # Compute mean squared value of gate outputs.
            # But we don't want large gradients to flow back to input features here.
            # So we scale down the bp stage of x with a grad scaler.
            # x: [n_exp, capacity, n_embd], gate_proj: [n_exp, n_embd, intermediate_size]
            # gate_out_cutoff: [n_exp, capacity, intermediate_size]
            # capacity: number of tokens sent to each expert.
            # NOTE: If some of x are padded zeros, the gate output losses of those elements would be 0.
            # The mean loss would be slightly smaller. So we should filter out those 
            # padded elements when computing the mean.
            gate_out_gs = torch.bmm(self.grad_scaler(x), self.gate_proj)
            gate_output_losses = (gate_out_gs ** 2).mean(dim=-1) # [n_exp, capacity]
            # Filter out zero elements.
            nonzero_mask = gate_output_losses > 0
            if nonzero_mask.sum() > 0:
                self.gate_output_loss = gate_output_losses[nonzero_mask].mean()
            else:
                self.gate_output_loss = torch.tensor(0.0, device=x.device)

        fc_out = torch.bmm(x, self.c_fc)
        x = self.act_fn(gate_out) * fc_out
        x = torch.bmm(x, self.c_proj)
        return x
    
class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        if getattr(config, 'use_qwen3_moe_mlp', False) and config.use_qwen3_moe_mlp:
            self.experts = Qwen3MLPExperts(config)
            self.use_qwen3_moe_mlp = True
        else:
            self.experts = MLPExperts(config)
            self.use_qwen3_moe_mlp = False

        self.n_exp = config.n_exp
        self.top_k = config.moe_top_k
        self.use_router_ortho_loss = config.use_router_ortho_loss
        self.router_ortho_neg_corr_weight = config.router_ortho_neg_corr_weight
        # Should always be True, i.e., always compute experts ortho loss for ablation study.
        # We just don't optimize it unless the weight is set > 0 in the config.
        self.use_experts_ortho_loss = config.use_experts_ortho_loss
        self.use_gate_output_loss = config.use_gate_output_loss
        # scale down gradients back to expert weights by 0.1 during router orthogonality loss computation.
        self.grad_scaler = gen_gradient_scaler(0.1) 

    def forward(self, x: torch.Tensor):
        # x: [64, 512, 512]
        B, T, C = x.size() # Keep track of original shape
        # --- Get routing information ---
        # Call the router with the ORIGINAL 3D tensor. The router will handle flattening internally
        # and return routing info shaped for a flattened list of tokens.
        expert_mask, router_probs, top_k_indices, rank = self.router(x)
        # expert_mask: [B*T, k, n_exp], router_probs: [B*T, k], etc.
        if self.training and self.use_router_ortho_loss:
            router_ortho_loss = self.compute_router_ortho_loss()
            MANAGER.add_router_ortho_loss(router_ortho_loss)
            # Always use gate diversity loss when using router orthogonality loss.
            projs_diversity_loss = self.compute_projs_diversity_loss()
            MANAGER.add_projs_diversity_loss(projs_diversity_loss)

        if self.training and self.use_experts_ortho_loss:
            experts_ortho_loss = self.compute_experts_ortho_loss()
            MANAGER.add_experts_ortho_loss(experts_ortho_loss)

        # Now, flatten the input tensor for the dispatch operation
        x_flat = x.view(B * T, C)

        # --- Dispatch tokens to experts (the "scatter" part) ---
        exp_capacity = self.router.get_capacity(B * T)
        expert_inputs = torch.zeros(self.n_exp, exp_capacity, C, dtype=x.dtype, device=x.device)

        # Get the indices for the valid assignments that are within capacity
        flat_top_k_indices = top_k_indices.view(-1)
        flat_rank = rank.view(-1)
        flat_token_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        valid_mask = flat_rank < exp_capacity
        valid_token_indices = flat_token_indices[valid_mask]
        valid_expert_indices = flat_top_k_indices[valid_mask]
        valid_ranks = flat_rank[valid_mask]

        # Use advanced indexing to place tokens from the flattened input into the expert buffer
        expert_inputs[valid_expert_indices, valid_ranks] = x_flat[valid_token_indices]

        # --- Run experts ---
        expert_outputs = self.experts(expert_inputs) # [n_exp, exp_capacity, C]

        # Only collect gate output loss after self.experts forward.
        if self.training and self.use_gate_output_loss:
            MANAGER.add_gate_output_loss(self.experts.gate_output_loss)
            self.experts.gate_output_loss = 0  # reset for next step

        # --- Combine expert outputs (the "gather" part) ---
        output_flat = torch.zeros_like(x_flat)

        # Retrieve the expert outputs using the same valid indices
        gated_expert_outputs = expert_outputs[valid_expert_indices, valid_ranks]

        # Filter router probabilities for valid assignments
        valid_router_probs = router_probs.view(-1)[valid_mask].unsqueeze(1)

        # Weight the expert outputs by the router probabilities
        weighted_outputs = gated_expert_outputs * valid_router_probs

        # Use scatter_add_ to combine outputs for tokens sent to multiple experts (k > 1)
        # This adds the weighted outputs back to their original token positions in the flattened output tensor.
        output_flat.scatter_add_(0, valid_token_indices.unsqueeze(1).expand_as(weighted_outputs), weighted_outputs)

        # Reshape output back to the original input shape
        return output_flat.view(B, T, C)

    def compute_router_ortho_loss(self):
        if not self.use_qwen3_moe_mlp:
            # Only apply orthogonality loss when using Qwen3-style MoE MLPs
            return 0
        else:
            # Compute orthogonality loss between router weight vectors and expert gate projection vectors
            router_weights = self.router.w_g.weight.unsqueeze(-1)  # [n_exp, n_embd, 1]
            # Totally cutting off gradients may not be optimal.
            # Scale down gradients to expert gate projection weights by 0.2  
            # allows adjusting expert weights slightly, without hurting representation learning too much.
            # gate_proj_weights: [n_exp, n_embd, intermediate_size]
            gate_proj_weights = self.grad_scaler(self.experts.gate_proj)  
            # ortho_losses: [n_exp, intermediate_size]
            ortho_losses = (router_weights * gate_proj_weights).sum(dim=1)
            ortho_losses_weights = torch.ones_like(ortho_losses)
            # Negative correlations could be downweighted by setting router_ortho_neg_corr_weight < 1.
            # But experiments seem to suggest that it's better to penalize negative correlations equally.
            ortho_losses_weights[ortho_losses < 0] = self.router_ortho_neg_corr_weight       
            # Square the dot products to penalize both positive and negative correlations
            ortho_losses = ortho_losses.square()
            # Change mean to sum, otherwise the loss is too small to have effect.
            # sum() is n_exp * intermediate_size times larger than mean()
            # n_exp = 128, intermediate_size = 2048, so the loss is 262144 times larger!!
            ortho_loss = (ortho_losses * ortho_losses_weights).sum()
            return ortho_loss

    # use_rand_estimate: speed up diversity loss computation with stochastic estimate.
    def compute_projs_diversity_loss(self, use_rand_estimate=True, num_rand_probes=1):
        loss = 0

        if not self.use_qwen3_moe_mlp:
            # Only apply orthogonality loss when using Qwen3-style MoE MLPs
            return loss

        for proj_name in ('gate_proj', 'c_fc'):
            # G: [n_exp, n_embd, intermediate_size]
            G = getattr(self.experts, proj_name)
            # Row-normalize: normalize each row vector over intermediate_size
            G = G / (G.norm(dim=2, keepdim=True) + 1e-12)
            E, D, F = G.size()  # n_exp, n_embd, intermediate_size

            if use_rand_estimate:
                # Stochastic Hutchinson trace/Frobenius estimator.
                # 2 probs are not accurate, but slightly faster than the exact method.
                # On the long term, it can still provide useful signal 
                # to improve diversity and suppress collapse.
                K = num_rand_probes
                # Z: [E, D, K]  (±1)
                Z = torch.empty((E, D, K), device=G.device, dtype=torch.int8).random_(2)
                Z = (Z * 2 - 1).to(G.dtype)
                # gt_z = G^T Z: [E, F, K]
                gt_z = torch.bmm(G.transpose(1, 2), Z)
                # Ggt_z = G gt_z: [E, D, K]
                Ggt_z = torch.bmm(G, gt_z)
                Az = Ggt_z - Z
                est_frob2 = Az.square().sum(dim=1).mean(dim=1)  # [E]  sum over D, mean over K
                row_sim_per_expert = est_frob2 / (D * D - D)
            else:
                # Batched Gram: [n_exp, n_embd, n_embd]
                # This computes cosine similarity between all pairs of row vectors of 
                # each expert's gate projection matrix.
                gram = torch.bmm(G, G.transpose(1, 2))
                # Zero out diagonal without materializing eye per expert
                gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))

                # Mean squared off-diagonal similarity per expert
                # Off-diagonal count = n_embd * n_embd - n_embd
                offdiag_sq_sum = gram.square().sum(dim=(1, 2))       # [n_exp]
                row_sim_per_expert = offdiag_sq_sum / (D * D - D)    # [n_exp]

            loss += row_sim_per_expert.mean()
            return loss
        
    # Compute orthogonality loss between expert weight matrices.
    # This is an ablation study of arXiv:2601.00457.
    def compute_experts_ortho_loss(self):
        if not self.use_qwen3_moe_mlp:
            return torch.tensor(0.0, device=self.experts.c_fc.device)

        W = self.experts.c_fc  # [n_exp, n_embd, 4*n_embd]
        n_exp = W.shape[0]
        if n_exp < 2:
            return W.new_zeros(())

        X = W.reshape(n_exp, -1).float()  # do math in fp32. long vector math is unstable in fp16/bf16.
        X = X / (X.norm(dim=1, keepdim=True) + 1e-12)  # normalize per expert

        G = X @ X.t()  # cosine-sim Gram matrix, diag ~ 1
        offdiag = torch.triu(G, diagonal=1)
        # penalize non-orthogonality without sign cancellation
        loss = (offdiag ** 2).mean()
        return loss.to(W.dtype)

class Block(nn.Module):

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True)
        if use_moe:
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(PreTrainedModel, GenerationMixin):
    config_class = GPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None

        if config.n_exp == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride> layers
            blocks = []
            for i in range(config.n_layer):
                # TODO: how to implement this?
                # should we change below to i + 1 ?
                use_moe = (i >= config.moe_start_layer) and ((i + 1) % config.stride == 0)
                blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = blocks,
            ln_f = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # optionall use switch transformer special init scheme for experts
        # See pg. 10 here: https://arxiv.org/abs/2101.03961
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Initialize weights using HF pattern
        self.post_init()
        
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print("number of active parameters (n_exp=%d, top_k=%d): %.2fM" % (
              config.n_exp, config.moe_top_k, self.get_num_active_params(config.n_exp, config.moe_top_k)/1e6))
    
    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, value):
        self.transformer.wte = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = 0
        # seen: avoid double-counting tied parameters.
        seen = set()
        for param in self.parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            n_params += param.numel()
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_active_params(self, n_exp, top_k):
        """
        Return the number of active parameters in the model.
        Active parameters are those that are used during a forward pass.
        In MoE models, only a subset of expert parameters are active per token.
        """
        n_params = 0
        # seen: avoid double-counting tied parameters.
        seen = set()
        for name, param in self.named_parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if 'experts' in name:
                n_params += param.numel() * top_k / n_exp
            else:
                # Non-expert parameters are always active
                n_params += param.numel()
        return n_params
    
    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim] 
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2*w_std,
                    b=2*w_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts) or isinstance(module, Qwen3MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2*c_fc_std,
                    b=2*c_fc_std,
                )
                if isinstance(module, Qwen3MLPExperts):
                    # also init gate_proj the same way as c_fc, as their shapes are the same
                    torch.nn.init.trunc_normal_(
                        module.gate_proj,
                        mean=0.0,
                        std=c_fc_std,
                        a=-2*c_fc_std,
                        b=2*c_fc_std,
                    )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2*c_proj_std,
                    b=2*c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)
                if isinstance(module, Qwen3MLPExperts):
                    torch.nn.init.normal_(module.gate_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        idx=None,  # Keep backward compatibility
        targets=None,  # Keep backward compatibility
        return_dict=None,
        **kwargs,
    ):
        # Handle backward compatibility: idx/targets or input_ids/labels
        if idx is not None:
            input_ids = idx
        if targets is not None:
            labels = targets
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True
        
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        router_ortho_loss = 0
        losses = { 'ntp_loss': 0,
                   'aux_loss': 0,
                   'router_z_loss': 0,
                   'router_ortho_loss': 0,
                   'experts_ortho_loss': 0, 
                   'gate_output_loss': 0,
                   'projs_diversity_loss': 0
                 }

        # Always compute logits for all positions (HuggingFace standard)
        logits = self.lm_head(x)
        
        if labels is not None:
            # Compute loss when labels are provided
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            losses['ntp_loss'] = loss.item()

            # add the auxiliary load balancing loss and router z loss to the main loss
            if self.config.n_exp > 1 and self.config.use_aux_loss:
                aux_loss = MANAGER.aggregate_aux_loss()
                loss += self.config.aux_loss_weight * aux_loss
                losses['aux_loss'] = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                MANAGER.reset_aux_loss()
            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                router_z_loss = MANAGER.aggregate_router_z_loss()
                loss += self.config.router_z_loss_weight * router_z_loss
                losses['router_z_loss'] = router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else router_z_loss
                MANAGER.reset_router_z_loss()
            if self.config.n_exp > 1 and self.config.use_router_ortho_loss:
                router_ortho_loss = MANAGER.aggregate_router_ortho_loss()
                loss += self.config.router_ortho_loss_weight * router_ortho_loss 
                losses['router_ortho_loss'] = router_ortho_loss.item() if isinstance(router_ortho_loss, torch.Tensor) else router_ortho_loss
                MANAGER.reset_router_ortho_loss()
                projs_diversity_loss = MANAGER.aggregate_projs_diversity_loss()
                loss += self.config.projs_diversity_loss_weight * projs_diversity_loss
                losses['projs_diversity_loss'] = projs_diversity_loss.item() if isinstance(projs_diversity_loss, torch.Tensor) else projs_diversity_loss
                MANAGER.reset_projs_diversity_loss()
            if self.config.n_exp > 1 and self.config.use_experts_ortho_loss:
                experts_ortho_loss = MANAGER.aggregate_experts_ortho_loss()
                loss += self.config.experts_ortho_loss_weight * experts_ortho_loss
                losses['experts_ortho_loss'] = experts_ortho_loss.item() if isinstance(experts_ortho_loss, torch.Tensor) else experts_ortho_loss
                MANAGER.reset_experts_ortho_loss()
            if self.config.n_exp > 1 and self.config.use_gate_output_loss:
                gate_output_loss = MANAGER.aggregate_gate_output_loss()
                loss += self.config.gate_output_loss_weight * gate_output_loss
                losses['gate_output_loss'] = gate_output_loss.item() if isinstance(gate_output_loss, torch.Tensor) else gate_output_loss
                MANAGER.reset_gate_output_loss()
        else:
            # No labels provided - inference mode
            loss = None

        if not return_dict:
            # Legacy return format: (logits, loss, losses)
            return logits, loss, losses
            
        # HuggingFace return format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def setup_optimizers(self, weight_decay, learning_rate, betas, device_type, embeddings_in_own_group=True):
        # TODO: add expert config
        decay_params = []
        nodecay_params = []
        embedding_params = []
        handled_params = set()
        # lm_head weight is tied to wte weight, so include it in embedding params
        embedding_suffixes = ('wte.weight', 'wpe.weight', 'lm_head.weight')

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in handled_params:
                continue
            handled_params.add(pid)

            is_bias = name.endswith('bias')
            is_embedding_weight = name.endswith(embedding_suffixes)

            if embeddings_in_own_group and is_embedding_weight:
                embedding_params.append(param)
            elif param.dim() >= 2 and not is_bias:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        if embeddings_in_own_group:
            optim_groups.append({'params': embedding_params, 'weight_decay': 0.0})

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        if embeddings_in_own_group:
            num_embedding_params = sum(p.numel() for p in embedding_params)
            print(f"num embedding parameter tensors: {len(embedding_params)}, with {num_embedding_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if use_fused:
            dtype_device_pairs = {
                (param.dtype, str(param.device))
                for group in optim_groups
                for param in group['params']
            }
            if len(dtype_device_pairs) > 1:
                combos = ', '.join(
                    f"{dtype} @ {device}" for dtype, device in sorted(
                        dtype_device_pairs, key=lambda x: (x[1], str(x[0]))
                    )
                )
                print(f"Disabling fused AdamW due to mixed parameter dtypes/devices: {combos}")
                use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    # TODO: integrate setup_optimizers() from nanochat, and remove the above configure_optimizers.
    # It create optim groups. Any parameters that is 2D will be in Muon and weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls decay, and all embeddings, biases and layernorms don't.
    # In the original nanoMoE code, embeddings are decayed which is not a good practice.
    # Otherwise, this is the same as nanoMoE's setup_optimizers().
    # add an extra check for "bias" string to account for bias terms in MoE layers
    def setup_optimizers2(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 4 groups (matrix, embedding, ln_f, position_embedding)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        position_embedding_params = list(self.transformer.wpe.parameters())
        # Don't include lm_head_params here, as those are tied with embedding_params.
        ln_f_params = list(self.transformer.ln_f.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) \
                        + len(position_embedding_params) + len(ln_f_params)
        # Create the AdamW optimizer for the embedding, ln_f, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=position_embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=ln_f_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of GPU bfloat16 -> fp32 accum peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # Determine the theoretical peak FLOPs of the current device using a simple lookup.
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()

            # Very small lookup table of common GPUs and their BF16/FP16 peak throughput (in FLOPs).
            # TODO: add more GPUs
            flops_table = {
                "3090": 71e12,   # RTX 3090
                "4090": 165e12,  # RTX 4090
                "l40s": 362e12,  # L40S
                "a100": 312e12,  # A100 80GB
                "h100": 990e12,  # H100
                "h200": 990e12,  # H200 (assumed same as H100 for BF16/FP16)
                "5070 ti": 176e12,  # RTX 5070 Ti
                "5080": 225e12,  # RTX 5080
                "b200": 2250e12,  # B200
                "rtx 6000 ada": 364e12,
                "rtx a6000": 155e12,   # dense tensor (BF16/FP16) approx; datasheet tensor is 309.7 TFLOPS with sparsity
            }

            # Pick the first entry whose key is a substring of the device name; fall back to 0.
            flops_promised = next((v for k, v in flops_table.items() if k in device_name), 0)
        else:
            # If running on CPU or an unknown accelerator, return -1 
            flops_promised = -1
        try:
            mfu = flops_achieved / flops_promised
        except:
            breakpoint()
        return mfu

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation."""
        return {"input_ids": input_ids}
    
    @torch.no_grad()
    def generate_legacy(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Legacy generation method for backward compatibility.
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            output = self(idx_cond, return_dict=True)
            logits = output.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Register the model with HuggingFace AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("nanomoe_gpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, GPT)
