"""
Prepare Swallow-Math v2 (stage3-qa) for pretraining.

Downloads all .jsonl files under stage3-qa/ from:
https://huggingface.co/datasets/tokyotech-llm/swallow-math-v2

Tokenizes with the GPT-2 tokenizer and writes monolithic .bin files.
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice

# ------------------------------------------
DATASET_NAME = "tokyotech-llm/swallow-math-v2"
DATASET_FILES = "stage3-qa/*.jsonl"
DATA_CACHE_DIR = os.path.dirname(__file__)


def iter_batches(dataset_iter, batch_size: int):
    while True:
        batch = list(islice(dataset_iter, batch_size))
        if not batch:
            break
        yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Swallow-Math v2 stage3-qa dataset")
    parser.add_argument("--size", type=int, default=50,
                        help="Target number of tokens in billions (default: 50)")
    parser.add_argument("--skip_docs", type=int, default=0,
                        help="Number of documents to skip (default: 0)")
    parser.add_argument("--skip_tokens", type=int, default=0,
                        help="Number of tokens in billions to skip (default: 0)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of documents to tokenize at once (default: 1000)")
    parser.add_argument("--text_key", type=str, default="text",
                        help="JSON field containing text (default: text)")
    parser.add_argument("--val_ratio", type=float, default=0.0005,
                        help="Validation split ratio (default: 0.0005)")
    args = parser.parse_args()

    target_tokens = args.size * 1_000_000_000
    if target_tokens <= 0:
        raise SystemExit("--size must be > 0")

    print(f"Loading dataset to collect ~{target_tokens:,} tokens ({args.size}B)...")
    dataset = load_dataset(
        DATASET_NAME,
        data_files=DATASET_FILES,
        split=None,
    )
    if hasattr(dataset, "values"):
        dataset = next(iter(dataset.values()))

    if args.skip_docs > 0:
        print(f"Skipping first {args.skip_docs:,} documents...")
        dataset = dataset.skip(args.skip_docs)

    enc = tiktoken.get_encoding("gpt2")

    dataset_iter = iter(dataset)

    # Skip tokens if specified
    skipped_examples = 0
    if args.skip_tokens > 0:
        skip_target = args.skip_tokens * 1_000_000_000
        print(f"Skipping first {args.skip_tokens}B tokens ({skip_target:,} tokens)...")
        tokens_skipped = 0
        pbar_skip = tqdm(total=skip_target, unit="tokens", unit_scale=True, desc="Skipping")

        for batch in iter_batches(dataset_iter, args.batch_size):
            batch_texts = [
                sample.get(args.text_key)
                for sample in batch
                if isinstance(sample.get(args.text_key), str)
            ]
            if not batch_texts:
                continue
            batch_ids = enc.encode_ordinary_batch(batch_texts)

            for ids in batch_ids:
                token_count = len(ids) + 1  # +1 for eot_token
                tokens_skipped += token_count
                skipped_examples += 1
                pbar_skip.update(token_count)

                if tokens_skipped >= skip_target:
                    break
            if tokens_skipped >= skip_target:
                break

        pbar_skip.close()
        print(f"Skipped {tokens_skipped:,} tokens ({skipped_examples:,} examples)")

    # First pass: write to temporary files while streaming
    temp_file = os.path.join(DATA_CACHE_DIR, "temp_all.bin")
    temp_idx_file = os.path.join(DATA_CACHE_DIR, "temp_all.idx")
    dtype = np.uint16
    arr = np.memmap(temp_file, dtype=dtype, mode="w+", shape=(target_tokens + 10_000_000,))

    total_tokens = 0
    num_examples = 0

    print(f"Streaming and tokenizing samples in batches of {args.batch_size}...")
    pbar = tqdm(total=target_tokens, unit="tokens", unit_scale=True)

    fidx = open(temp_idx_file, "wb")

    for batch in iter_batches(dataset_iter, args.batch_size):
        batch_texts = [
            sample.get(args.text_key)
            for sample in batch
            if isinstance(sample.get(args.text_key), str)
        ]
        if not batch_texts:
            continue

        batch_ids = enc.encode_ordinary_batch(batch_texts)

        for ids in batch_ids:
            ids = list(ids)
            ids.append(enc.eot_token)
            num_examples += 1

            token_count = len(ids)
            remaining = arr.shape[0] - total_tokens
            if token_count > remaining:
                ids = ids[:remaining]
                token_count = len(ids)

            arr[total_tokens:total_tokens + token_count] = ids
            np.asarray([total_tokens, token_count], dtype=np.int64).tofile(fidx)
            total_tokens += token_count
            pbar.update(token_count)

            if total_tokens >= target_tokens:
                break

        if total_tokens >= target_tokens:
            break

    pbar.close()
    fidx.close()

    print(f"Processed {num_examples:,} examples to get {total_tokens:,} tokens")

    # Trim to actual size
    arr.flush()
    del arr

    with open(temp_file, "r+b") as f:
        f.truncate(total_tokens * np.dtype(dtype).itemsize)

    if total_tokens == 0:
        raise SystemExit("No tokens were collected. Check --text_key or dataset availability.")

    arr = np.memmap(temp_file, dtype=dtype, mode="r", shape=(total_tokens,))
    print(f"Collected {total_tokens:,} tokens")

    val_target = int(total_tokens * args.val_ratio)
    train_target = total_tokens - val_target

    # Align split to sample boundaries using the temp idx
    idx_pair_size = np.dtype(np.int64).itemsize * 2
    num_samples = os.path.getsize(temp_idx_file) // idx_pair_size
    idx_arr = np.memmap(temp_idx_file, dtype=np.int64, mode="r", shape=(num_samples, 2))
    train_samples = 0
    train_size = 0
    for i in range(num_samples):
        length = int(idx_arr[i, 1])
        if train_size + length > train_target:
            break
        train_size += length
        train_samples += 1
    val_size = total_tokens - train_size

    print(f"Creating train/val split ({train_size:,} train, {val_size:,} val)...")

    suffix = f"-skip{args.skip_tokens}B" if args.skip_tokens > 0 else ""
    val_file = os.path.join(DATA_CACHE_DIR, f"val-{args.size}B{suffix}.bin")
    val_idx_file = os.path.join(DATA_CACHE_DIR, f"val-{args.size}B{suffix}.idx")
    val_arr = np.memmap(val_file, dtype=dtype, mode="w+", shape=(val_size,))

    print(f"Writing val-{args.size}B{suffix}.bin...")
    batch_size = 10_000_000
    for i in tqdm(range(0, val_size, batch_size)):
        end = min(i + batch_size, val_size)
        val_arr[i:end] = arr[train_size + i:train_size + end]
    val_arr.flush()
    del val_arr

    # Write val idx (offsets relative to val file)
    with open(val_idx_file, "wb") as fval_idx:
        for i in range(train_samples, num_samples):
            start = int(idx_arr[i, 0]) - train_size
            length = int(idx_arr[i, 1])
            np.asarray([start, length], dtype=np.int64).tofile(fval_idx)

    print(f"Creating train-{args.size}B{suffix}.bin...")
    del arr
    train_file = os.path.join(DATA_CACHE_DIR, f"train-{args.size}B{suffix}.bin")
    train_idx_file = os.path.join(DATA_CACHE_DIR, f"train-{args.size}B{suffix}.idx")
    with open(temp_file, "r+b") as f:
        f.truncate(train_size * np.dtype(dtype).itemsize)
    os.rename(temp_file, train_file)

    # Write train idx
    with open(train_idx_file, "wb") as ftrain_idx:
        for i in range(train_samples):
            np.asarray(idx_arr[i], dtype=np.int64).tofile(ftrain_idx)

    del idx_arr
    if os.path.exists(temp_idx_file):
        os.remove(temp_idx_file)

    print(
        f"Done! train-{args.size}B{suffix}.bin: {train_size:,} tokens, "
        f"val-{args.size}B{suffix}.bin: {val_size:,} tokens"
    )
