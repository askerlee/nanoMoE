"""
Revised from:
https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves to monolithic .bin files.
Run simply as:
$ python prepare.py
Will save train.bin and val.bin to the local directory "edu_fineweb10B".
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
remote_name = "sample-10BT"

# number of workers in .map() call
num_proc = max(1, os.cpu_count()//2)

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.dirname(__file__)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # download the dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    
    # create a small validation split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    
    # tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(DATA_CACHE_DIR, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        