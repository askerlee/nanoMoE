"""
Revised from: 
https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves to monolithic . bin files. 
Run with custom size (in billions of tokens):
$ python prepare.py --size 30
$ python prepare.py --size 100
Will save train-{size}B.bin and val-{size}B.bin to the same directory as this script.
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice

# ------------------------------------------
remote_name = "sample-100BT"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.dirname(__file__)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Prepare FineWeb-Edu dataset with custom token count')
    parser.add_argument('--size', type=int, default=50, 
                        help='Target number of tokens in billions (default: 50)')
    parser.add_argument('--skip_docs', type=int, default=0,
                        help='Number of documents to skip (default: 0)')
    parser.add_argument('--skip_tokens', type=int, default=0,
                        help='Number of tokens in billions to skip (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Number of documents to tokenize at once (default: 1000)')
    args = parser.parse_args()
    
    # Target number of tokens
    target_tokens = args.size * 1_000_000_000
    
    print(f"Loading dataset to collect ~{target_tokens:,} tokens ({args.size}B)...")
    # NOTE: leave enough space in ~/.cache/huggingface/datasets for the dataset download (~200GB).
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    
    # Skip documents if specified
    if args.skip_docs > 0:
        print(f"Skipping first {args.skip_docs:,} documents...")
        dataset = dataset.skip(args.skip_docs)
    
    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Skip tokens if specified
    dataset_iter = iter(dataset)
    skipped_examples = 0
    if args.skip_tokens > 0:
        skip_target = args.skip_tokens * 1_000_000_000
        print(f"Skipping first {args.skip_tokens}B tokens ({skip_target:,} tokens)...")
        tokens_skipped = 0
        pbar_skip = tqdm(total=skip_target, unit='tokens', unit_scale=True, desc="Skipping")
        
        while tokens_skipped < skip_target:
            batch = list(islice(dataset_iter, args.batch_size))
            if not batch:
                print(f"Warning: Reached end of dataset after skipping {tokens_skipped:,} tokens")
                break
            
            batch_texts = [sample['text'] for sample in batch]
            batch_ids = enc.encode_ordinary_batch(batch_texts)
            
            for ids in batch_ids:
                token_count = len(ids) + 1  # +1 for eot_token
                tokens_skipped += token_count
                skipped_examples += 1
                pbar_skip.update(token_count)
                
                if tokens_skipped >= skip_target:
                    break
        pbar_skip.close()
        print(f"Skipped {tokens_skipped:,} tokens ({skipped_examples:,} examples)")
    
    # First pass: write to temporary file while streaming
    temp_file = os.path.join(DATA_CACHE_DIR, 'temp_all.bin')
    dtype = np.uint16
    
    arr = np.memmap(temp_file, dtype=dtype, mode='w+', shape=(target_tokens + 10_000_000,))
    
    total_tokens = 0
    num_examples = 0
    
    print(f"Streaming and tokenizing samples in batches of {args.batch_size}...")
    # Each sample is a single document in the original dataset with varying lengths.
    pbar = tqdm(total=target_tokens, unit='tokens', unit_scale=True)
    
    while total_tokens < target_tokens:
        # Get a batch of documents
        batch = list(islice(dataset_iter, args.batch_size))
        if not batch:
            break
        
        # Tokenize the entire batch
        batch_texts = [sample['text'] for sample in batch]
        batch_ids = enc.encode_ordinary_batch(batch_texts)
        
        # Process each tokenized document
        for ids in batch_ids:
            ids = list(ids)
            ids.append(enc.eot_token)
            num_examples += 1
            
            # Write directly to memmap
            token_count = len(ids)
            remaining = arr.shape[0] - total_tokens
            if token_count > remaining:
                ids = ids[:remaining]
                token_count = len(ids)

            arr[total_tokens : total_tokens + token_count] = ids
            total_tokens += token_count
            pbar.update(token_count)
            
            # Since the samples have varying lengths, we check if we've reached target.
            if total_tokens >= target_tokens:
                break
    pbar.close()
    
    print(f"Processed {num_examples:,} examples to get {total_tokens:,} tokens")

    # Trim to actual size
    arr.flush()
    del arr
    
    # Truncate file to actual length to save disk space
    with open(temp_file, 'r+b') as f:
        f.truncate(total_tokens * np.dtype(dtype).itemsize)
    
    arr = np.memmap(temp_file, dtype=dtype, mode='r', shape=(total_tokens,))
    print(f"Collected {total_tokens: ,} tokens")
    
    # Create train/val split
    val_size = int(total_tokens * 0.0005)
    train_size = total_tokens - val_size
    
    print(f"Creating train/val split ({train_size:,} train, {val_size: ,} val)...")
    
    # Write val file first (copy the tail)
    suffix = f'-skip{args.skip_tokens}B' if args.skip_tokens > 0 else ''
    val_file = os.path.join(DATA_CACHE_DIR, f'val-{args.size}B{suffix}.bin')
    val_arr = np.memmap(val_file, dtype=dtype, mode='w+', shape=(val_size,))
    
    print(f"Writing val-{args.size}B{suffix}.bin...")
    batch_size = 10_000_000
    for i in tqdm(range(0, val_size, batch_size)):
        end = min(i + batch_size, val_size)
        val_arr[i:end] = arr[train_size + i:train_size + end]
    val_arr. flush()
    del val_arr
    
    # Truncate temp file to train size and rename it
    print(f"Creating train-{args.size}B{suffix}.bin...")
    del arr
    train_file = os.path.join(DATA_CACHE_DIR, f'train-{args.size}B{suffix}.bin')
    with open(temp_file, 'r+b') as f:
        f.truncate(train_size * np.dtype(dtype).itemsize)
    os.rename(temp_file, train_file)

    print(f"Done! train-{args.size}B{suffix}.bin: {train_size:,} tokens, val-{args.size}B{suffix}.bin: {val_size:,} tokens")
    