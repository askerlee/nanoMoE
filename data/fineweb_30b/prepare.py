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
    
    # Estimate tokens per sample to pre-allocate (average ~500 tokens per sample)
    # Over-allocate to avoid resizing, we'll trim later
    # average 500: FineWeb-Edu contains web documents with varying lengths.
    # Some might be 100 tokens, others 2000+ tokens. 
    # Claude guessed ~500 as a ballpark average.
    estimated_samples = int(target_tokens / 500 * 1.1)
    arr = np.memmap(temp_file, dtype=dtype, mode='w+', shape=(target_tokens + 10_000_000,))
    
    idx = 0
    total_tokens = 0
    
    print("Streaming and tokenizing samples...")
    # Each sample is a single document in the original dataset with varying lengths.
    for sample in tqdm(dataset, total=estimated_samples):
        ids = enc.encode_ordinary(sample['text'])
        ids.append(enc.eot_token)
        
        # Write directly to memmap
        token_count = len(ids)
        arr[idx : idx + token_count] = ids
        idx += token_count
        total_tokens += token_count
        
        # Since the samples have varying lengths, we check if we've reached target.
        if total_tokens >= target_tokens:
            break
    
    # Trim to actual size
    arr.flush()
    del arr
    
    # Resize to actual length
    arr = np.memmap(temp_file, dtype=dtype, mode='r+', shape=(idx,))
    arr.flush()
    print(f"Collected {total_tokens:,} tokens")
    
    # Create train/val split (shuffle in chunks to avoid loading all into RAM)
    val_size = int(idx * 0.0005)
    train_size = idx - val_size
    
    print(f"Creating train/val split ({train_size:,} train, {val_size:,} val)...")
    
    # Shuffle indices
    np.random.seed(2357)
    indices = np.arange(idx)
    # Chunk-based shuffle to avoid memory issues with huge arrays
    chunk_size = 100_000_000
    for i in range(0, idx, chunk_size):
        end = min(i + chunk_size, idx)
        np.random.shuffle(indices[i:end])
    
    # Write train and val files
    train_file = os.path.join(DATA_CACHE_DIR, 'train.bin')
    val_file = os.path.join(DATA_CACHE_DIR, 'val.bin')
    
    train_arr = np.memmap(train_file, dtype=dtype, mode='w+', shape=(train_size,))
    val_arr = np.memmap(val_file, dtype=dtype, mode='w+', shape=(val_size,))
    
    # Copy to train
    print("Writing train.bin...")
    batch_size = 10_000_000
    for i in tqdm(range(0, train_size, batch_size)):
        end = min(i + batch_size, train_size)
        train_arr[i:end] = arr[indices[i:end]]
    train_arr.flush()
    
    # Copy to val
    print("Writing val.bin...")
    for i in tqdm(range(0, val_size, batch_size)):
        end = min(i + batch_size, val_size)
        val_arr[i:end] = arr[indices[train_size + i:train_size + end]]
    val_arr.flush()
    
    # Cleanup
    del arr, train_arr, val_arr
    os.remove(temp_file)
    
    print(f"Done! train.bin: {train_size:,} tokens, val.bin: {val_size:,} tokens")
