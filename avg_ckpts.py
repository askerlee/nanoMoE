import os, json, glob
from collections import OrderedDict
import torch

def _find_hf_weight_files(ckpt_dir: str):
    """
    Returns either:
      ("safetensors_single", path)
      ("bin_single", path)
      ("safetensors_sharded", index_json_path)
      ("bin_sharded", index_json_path)
    """
    # Single-file cases
    st = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(st):
        return ("safetensors_single", st)

    bn = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(bn):
        return ("bin_single", bn)

    # Sharded cases (index.json)
    st_idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(st_idx):
        return ("safetensors_sharded", st_idx)

    bn_idx = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(bn_idx):
        return ("bin_sharded", bn_idx)

    # Fallback: try to locate any *index.json
    idxs = glob.glob(os.path.join(ckpt_dir, "*.index.json"))
    if idxs:
        p = idxs[0]
        if "safetensors" in os.path.basename(p):
            return ("safetensors_sharded", p)
        return ("bin_sharded", p)

    raise FileNotFoundError(f"Could not find HF weights in: {ckpt_dir}")

def _load_index(index_json_path: str):
    with open(index_json_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    # unique shard files
    shard_files = sorted(set(weight_map.values()))
    return weight_map, shard_files

def _iter_safetensors_shard_tensors(shard_path: str):
    # Stream tensors one-by-one via memory mapping (low peak RAM)
    from safetensors import safe_open
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)

def _iter_bin_shard_tensors(shard_path: str):
    # torch.load loads the whole shard dict into RAM
    sd = torch.load(shard_path, map_location="cpu")
    for k, v in sd.items():
        if torch.is_tensor(v):
            yield k, v

@torch.no_grad()
def average_hf_checkpoints(
    ckpt_dirs,
    out_path="avg_state_dict.pt",
    dtype=torch.float32,   # accumulator dtype
    strict_keys=True,      # require identical keys across ckpts
):
    assert len(ckpt_dirs) > 0

    avg = OrderedDict()
    seen_keys = None

    for i, ckpt_dir in enumerate(ckpt_dirs, start=1):
        kind, path = _find_hf_weight_files(ckpt_dir)

        # Collect tensors for this checkpoint as an iterator of (name, tensor)
        if kind == "safetensors_single":
            tensors = _iter_safetensors_shard_tensors(path)
        elif kind == "bin_single":
            tensors = _iter_bin_shard_tensors(path)
        elif kind in ("safetensors_sharded", "bin_sharded"):
            weight_map, shard_files = _load_index(path)
            # We'll iterate shards; each shard yields its own keys.
            shard_paths = [os.path.join(ckpt_dir, sf) for sf in shard_files]

            def _iter_all():
                for sp in shard_paths:
                    if kind == "safetensors_sharded":
                        yield from _iter_safetensors_shard_tensors(sp)
                    else:
                        yield from _iter_bin_shard_tensors(sp)
            tensors = _iter_all()
        else:
            raise RuntimeError(f"Unknown kind: {kind}")

        # For strict key checking, gather this ckpt's keyset (cheap for safetensors, costly for bin)
        # We do a one-pass approach: record keys we encounter.
        cur_keys = set()

        for k, v in tensors:
            cur_keys.add(k)
            if not torch.is_tensor(v):
                continue

            v = v.detach().to(dtype=dtype, device="cpu")

            if k not in avg:
                # first checkpoint initializes the running mean
                if i == 1:
                    avg[k] = v.clone()
                else:
                    if strict_keys:
                        raise KeyError(f"Key {k} appeared in {ckpt_dir} but not in earlier checkpoints")
                    else:
                        # skip keys that aren't common
                        continue
            else:
                # running mean: avg <- avg + (v - avg)/i
                avg[k].add_((v - avg[k]) / i)

        if strict_keys:
            if seen_keys is None:
                seen_keys = cur_keys
            else:
                missing = seen_keys - cur_keys
                extra = cur_keys - seen_keys
                if missing or extra:
                    raise KeyError(
                        f"Key mismatch in {ckpt_dir}\n"
                        f"Missing: {sorted(list(missing))[:10]} ...\n"
                        f"Extra: {sorted(list(extra))[:10]} ..."
                    )

        print(f"Averaged {i}/{len(ckpt_dirs)}: {ckpt_dir}")

    torch.save(avg, out_path)
    print(f"Saved averaged state_dict to: {out_path}")
    return out_path


group = "lossless" # loss, lossless
if group == "loss":
    ckpt_dirs = ["nanoMoE_out0/sb0r-950", "nanoMoE_out0/sb0r-1000"]
    base_model_dir = "nanoMoE_out0/sb0r-1000"
    avg_sd_path = "nanoMoE_out0/sb0r-avg.pt"
    out_dir = "nanoMoE_out0/sb0r-avg"
else:
    ckpt_dirs = ["nanoMoE_out1/sb1r-950", "nanoMoE_out1/sb1r-1000"]
    base_model_dir = "nanoMoE_out1/sb1r-1000"
    avg_sd_path = "nanoMoE_out1/sb1r-avg.pt"
    out_dir = "nanoMoE_out1/sb1r-avg"

#average_hf_checkpoints(ckpt_dirs, out_path="nanoMoE_out1/sb1r-avg.pt")

from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

sd = torch.load(avg_sd_path, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)

# Important for tied weights in some models
if hasattr(model, "tie_weights"):
    model.tie_weights()

model.save_pretrained(out_dir, safe_serialization=True)  # will shard automatically if large
