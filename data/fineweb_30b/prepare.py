"""
Revised from:
https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves to monolithic .bin files.
Run simply as:
$ python prepare.py
Will save train.bin and val.bin to the same directory as this script.
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
remote_name = "sample-100BT"
# Target number of tokens (30B)
target_tokens = 30_000_000_000

# number of workers in .map() call
num_proc = max(1, os.cpu_count()//2)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.dirname(__file__)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Stream the dataset and write directly to disk to avoid loading 30B tokens in RAM
    print(f"Streaming dataset to collect ~{target_tokens:,} tokens...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
    
    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # First pass: write to temporary file while streaming
    temp_file = os.path.join(DATA_CACHE_DIR, 'temp_all.bin')
    dtype = np.uint16
    
    arr = np.memmap(temp_file, dtype=dtype, mode='w+', shape=(target_tokens + 10_000_000,))
    
    idx = 0
    total_tokens = 0
    
    print("Streaming and tokenizing samples...")
    # Each sample is a single document in the original dataset with varying lengths.
    pbar = tqdm(total=target_tokens, unit='tokens', unit_scale=True)
    for sample in dataset:
        ids = enc.encode_ordinary(sample['text'])
        ids.append(enc.eot_token)
        
        # Write directly to memmap
        token_count = len(ids)
        arr[idx : idx + token_count] = ids
        idx += token_count
        total_tokens += token_count
        pbar.update(token_count)
        
        # Since the samples have varying lengths, we check if we've reached target.
        if total_tokens >= target_tokens:
            break
    pbar.close()
    
    # Trim to actual size
    arr.flush()
    del arr
    
    # Truncate file to actual length to save disk space
    with open(temp_file, 'r+b') as f:
        f.truncate(idx * np.dtype(dtype).itemsize)
    
    arr = np.memmap(temp_file, dtype=dtype, mode='r', shape=(idx,))
    print(f"Collected {total_tokens:,} tokens")
    
    # Create train/val split
    val_size = int(idx * 0.0005)
    train_size = idx - val_size
    
    print(f"Creating train/val split ({train_size:,} train, {val_size:,} val)...")
    
    # Write val file first (copy the tail)
    val_file = os.path.join(DATA_CACHE_DIR, 'val.bin')
    val_arr = np.memmap(val_file, dtype=dtype, mode='w+', shape=(val_size,))
    
    print("Writing val.bin...")
    batch_size = 10_000_000
    for i in tqdm(range(0, val_size, batch_size)):
        end = min(i + batch_size, val_size)
        val_arr[i:end] = arr[train_size + i:train_size + end]
    val_arr.flush()
    del val_arr
    
    # Truncate temp file to train size and rename it
    print("Creating train.bin...")
    del arr
    train_file = os.path.join(DATA_CACHE_DIR, 'train.bin')
    with open(temp_file, 'r+b') as f:
        f.truncate(train_size * np.dtype(dtype).itemsize)
    os.rename(temp_file, train_file)

    print(f"Done! train.bin: {train_size:,} tokens, val.bin: {val_size:,} tokens")
