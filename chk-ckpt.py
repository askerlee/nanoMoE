#!/usr/bin/env python3
"""
check_ckpt_adamw.py

Usage examples:

# 1) Just inspect training_state/optimizer state (no GPU needed)
python check_ckpt_adamw.py --ckpt-dir out/24980-150

# 2) Full check: load model + optimizer + run one dummy step on CUDA (will reproduce fused-adam issues)
python check_ckpt_adamw.py --ckpt-dir out/24980-150 --device cuda --test-step

# 3) Same as (2) but force fused=False (useful to isolate fused-only issues)
python check_ckpt_adamw.py --ckpt-dir out/24980-150 --device cuda --test-step --disable-fused
"""

import argparse
import os
import sys
import traceback
from typing import Any, Dict, Tuple, Optional

import torch

# IMPORTANT: adjust import path if needed (same dir as your train.py/modeling file)
from modeling_nanomoe_gpt import GPT


def _pick(cfg: Any, keys, default=None):
    """Pick first available key from dict-like or attribute-like cfg."""
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            return cfg[k]
        if hasattr(cfg, k):
            return getattr(cfg, k)
    return default


def load_training_state(ckpt_dir: str, training_state_path: Optional[str] = None) -> Dict[str, Any]:
    if training_state_path is None:
        training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    if not os.path.isfile(training_state_path):
        raise FileNotFoundError(f"training_state not found: {training_state_path}")
    ts = torch.load(training_state_path, map_location="cpu", weights_only=False)
    if not isinstance(ts, dict):
        raise TypeError(f"training_state is not a dict, got {type(ts)}")
    return ts


def build_adamw_like_nanomoe(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    betas: Tuple[float, float],
    embeddings_in_own_group: bool,
    fused: Optional[bool],
) -> torch.optim.Optimizer:

    decay_params = []
    nodecay_params = []
    embedding_params = []
    handled_params = set()
    # lm_head weight is tied to wte weight, so include it in embedding params
    embedding_suffixes = ('wte.weight', 'wpe.weight', 'lm_head.weight')

    for name, param in model.named_parameters():
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

    extra_args = {}
    sig = torch.optim.AdamW.__init__.__code__.co_varnames
    if fused is not None:
        # only pass fused if AdamW supports it
        if "fused" in sig:
            extra_args["fused"] = bool(fused)
    else:
        # default: let pytorch decide (typically fused=True on cuda in your code)
        pass

    opt = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
    return opt


def summarize_training_state(ts: Dict[str, Any]) -> None:
    print("=== training_state summary ===")
    for k in ["global_iter", "persist_global_iter", "eval_count", "best_val_loss", "epoch", "batch_idx"]:
        if k in ts:
            print(f"{k}: {ts[k]}")
    cfg = ts.get("config", None)
    if cfg is None:
        print("config: <missing>")
    else:
        wd = _pick(cfg, ["weight_decay"], None)
        lr = _pick(cfg, ["learning_rate", "lr"], None)
        b1 = _pick(cfg, ["beta1"], None)
        b2 = _pick(cfg, ["beta2"], None)
        print(f"config(weight_decay={wd}, learning_rate={lr}, beta1={b1}, beta2={b2})")
    osd = ts.get("optimizer_state_dict", None)
    if isinstance(osd, list):
        print(f"optimizer_state_dict: list[{len(osd)}]")
    elif isinstance(osd, dict):
        print("optimizer_state_dict: dict")
    else:
        print(f"optimizer_state_dict: {type(osd)}")
    print("==============================\n")


def _get_optimizer_state_dict(ts: Dict[str, Any]) -> Dict[str, Any]:
    osd = ts.get("optimizer_state_dict", None)
    if osd is None:
        raise KeyError("training_state has no 'optimizer_state_dict'")
    if isinstance(osd, list):
        if len(osd) != 1:
            raise ValueError(f"optimizer_state_dict is a list of length {len(osd)}; this script expects 1 optimizer.")
        osd = osd[0]
    if not isinstance(osd, dict):
        raise TypeError(f"optimizer_state_dict must be dict, got {type(osd)}")
    return osd


def check_optimizer_state_internal(osd: Dict[str, Any]) -> int:
    """
    Lightweight check that does NOT require model:
    - state entries exist and are tensors where expected
    - exp_avg/exp_avg_sq shapes match each other
    - report dtypes/devices seen in checkpoint
    """
    print("=== optimizer_state_dict lightweight checks ===")
    state = osd.get("state", None)
    if not isinstance(state, dict):
        print("FAIL: optimizer_state_dict['state'] is missing or not a dict")
        return 1

    dtypes = {}
    devices = {}
    bad = 0
    checked = 0

    for pid, st in state.items():
        if not isinstance(st, dict):
            continue
        ea = st.get("exp_avg", None)
        eas = st.get("exp_avg_sq", None)
        step = st.get("step", None)

        if torch.is_tensor(ea):
            dtypes[str(ea.dtype)] = dtypes.get(str(ea.dtype), 0) + 1
            devices[str(ea.device)] = devices.get(str(ea.device), 0) + 1
        if torch.is_tensor(eas):
            dtypes[str(eas.dtype)] = dtypes.get(str(eas.dtype), 0) + 1
            devices[str(eas.device)] = devices.get(str(eas.device), 0) + 1

        if torch.is_tensor(ea) and torch.is_tensor(eas):
            checked += 1
            if ea.shape != eas.shape:
                bad += 1
                if bad <= 5:
                    print(f"BAD: pid={pid} exp_avg shape {tuple(ea.shape)} != exp_avg_sq shape {tuple(eas.shape)}")

        # step can be int/float or tensor depending on how it was saved
        if step is not None and not (torch.is_tensor(step) or isinstance(step, (int, float))):
            bad += 1
            if bad <= 5:
                print(f"BAD: pid={pid} step has weird type {type(step)}")

    print(f"Checked {checked} state entries with both exp_avg + exp_avg_sq.")
    print("Dtypes seen (from exp buffers):", dtypes)
    print("Devices seen (from exp buffers):", devices)
    if bad == 0:
        print("OK: No obvious internal corruption found.\n")
    else:
        print(f"FAIL: Found {bad} issues in internal optimizer state.\n")
    return 0 if bad == 0 else 2


def check_state_vs_model_and_optional_step(
    ckpt_dir: str,
    ts: Dict[str, Any],
    device: str,
    embeddings_in_own_group: bool,
    disable_fused: bool,
    test_step: bool,
    batch_size: int,
    seq_len: int,
) -> int:
    cfg = ts.get("config", {})
    weight_decay = float(_pick(cfg, ["weight_decay"], 0.0))
    lr = float(_pick(cfg, ["learning_rate", "lr"], 1e-4))
    beta1 = float(_pick(cfg, ["beta1"], 0.9))
    beta2 = float(_pick(cfg, ["beta2"], 0.95))
    betas = (beta1, beta2)

    print("=== loading model ===")
    model = GPT.from_pretrained(ckpt_dir)
    model.to(device)
    model.train()
    print("Model loaded.")

    fused = None
    if device.startswith("cuda"):
        fused = False if disable_fused else True
    else:
        fused = False  # fused doesn't apply on CPU

    print("=== building optimizer ===")
    opt = build_adamw_like_nanomoe(
        model=model,
        weight_decay=weight_decay,
        lr=lr,
        betas=betas,
        embeddings_in_own_group=embeddings_in_own_group,
        fused=fused,
    )
    print("Optimizer built.")

    osd = _get_optimizer_state_dict(ts)
    print("=== loading optimizer state_dict ===")
    try:
        opt.load_state_dict(osd)
    except Exception as e:
        print("FAIL: optimizer.load_state_dict() failed.")
        print("Exception:", repr(e))
        traceback.print_exc()
        return 3
    print("Optimizer state loaded.\n")

    # Check state tensors vs params
    print("=== checking optimizer state buffers vs model params ===")
    bad = 0
    checked = 0
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            st = opt.state.get(p, None)
            if not st:
                continue
            for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if k in st and torch.is_tensor(st[k]):
                    t = st[k]
                    checked += 1
                    if t.device != p.device or t.dtype != p.dtype:
                        bad += 1
                        if bad <= 10:
                            print(f"BAD: group {gi} {k} device/dtype mismatch: "
                                  f"{t.device}/{t.dtype} vs param {p.device}/{p.dtype}")
                    if t.shape != p.shape:
                        bad += 1
                        if bad <= 10:
                            print(f"BAD: group {gi} {k} shape mismatch: {tuple(t.shape)} vs param {tuple(p.shape)}")
                    if t.layout != torch.strided:
                        bad += 1
                        if bad <= 10:
                            print(f"BAD: group {gi} {k} layout is {t.layout}, expected strided")
    print(f"Checked {checked} state tensors against params.")
    if bad == 0:
        print("OK: state buffers match param dtype/device/shape/layout.\n")
    else:
        print(f"FAIL: found {bad} state-vs-param issues.\n")
        # still continue; test_step may reproduce exact fused failure

    if not test_step:
        return 0 if bad == 0 else 2

    if not device.startswith("cuda"):
        print("NOTE: --test-step on CPU will not exercise fused AdamW. Use --device cuda for the real repro.\n")

    # One dummy forward/backward/step
    print("=== running one dummy step ===")
    vocab_size = int(model.config.vocab_size)
    max_t = int(model.config.block_size)
    t = min(seq_len, max_t)

    idx = torch.randint(0, vocab_size, (batch_size, t), device=device, dtype=torch.long)

    opt.zero_grad(set_to_none=True)
    try:
        out = model(input_ids=idx, labels=idx)
        loss = out.loss
        if loss is None:
            raise RuntimeError("Model returned loss=None (labels not hooked up?)")
        loss.backward()
        opt.step()
    except Exception as e:
        print("FAIL: dummy step failed (this should catch your fused AdamW error in isolation).")
        print("Exception:", repr(e))
        traceback.print_exc()
        return 4

    print("OK: dummy step succeeded.\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="Checkpoint directory that contains training_state.pt and HF model files.")
    ap.add_argument("--training-state", default=None, help="Optional path to training_state.pt (defaults to ckpt-dir/training_state.pt)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Where to load model/optimizer for deep checks.")
    ap.add_argument("--embeddings-in-own-group", action="store_true", help="Match embeddings_in_own_group=True grouping (default).")
    ap.add_argument("--no-embeddings-in-own-group", dest="embeddings_in_own_group", action="store_false",
                    help="Use embeddings_in_own_group=False grouping.")
    ap.set_defaults(embeddings_in_own_group=True)
    ap.add_argument("--disable-fused", action="store_true", help="Force AdamW fused=False even on CUDA.")
    ap.add_argument("--test-step", action="store_true", help="Run one dummy forward/backward/step to reproduce fused issues.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available.")
        return 10

    ts = load_training_state(args.ckpt_dir, args.training_state)
    summarize_training_state(ts)

    osd = _get_optimizer_state_dict(ts)
    rc1 = check_optimizer_state_internal(osd)

    # Deep checks require model files
    rc2 = check_state_vs_model_and_optional_step(
        ckpt_dir=args.ckpt_dir,
        ts=ts,
        device=args.device,
        embeddings_in_own_group=args.embeddings_in_own_group,
        disable_fused=args.disable_fused,
        test_step=args.test_step,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    rc = max(rc1, rc2)
    print(f"Exit code: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
