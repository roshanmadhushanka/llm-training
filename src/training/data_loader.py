"""
Data loading - turning a big array of token ids into training batches.

We store our dataset as a flat binary file of uint16 token ids (`train.bin`
and `val.bin`, produced by ``src/data/prepare_shakespeare.py``). To build one
training example we:

  1. Pick a random start position in the data.
  2. Take `block_size` tokens starting there  -> this is the input  x.
  3. Take the SAME slice shifted right by one  -> this is the target y.

So for every position the model sees, the "answer" is simply the next token.
That single, simple objective - "predict the next token" - is all it takes to
train a language model.

    data:  [ ... A  B  C  D  E ... ]
    x:           A  B  C  D
    y:           B  C  D  E      (x shifted by one)
"""

import os

import numpy as np
import torch


class DataLoader:
    def __init__(self, data_dir: str, block_size: int, batch_size: int,
                 device, pin_memory: bool = False):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device          # torch.device or DirectML device
        self.pin_memory = pin_memory  # only useful for CUDA

    def _load(self, split: str) -> np.memmap:
        # np.memmap reads the file lazily from disk instead of loading it all
        # into RAM. We re-open it every batch to avoid a known memory leak.
        filename = "train.bin" if split == "train" else "val.bin"
        return np.memmap(os.path.join(self.data_dir, filename), dtype=np.uint16, mode="r")

    def get_batch(self, split: str):
        data = self._load(split)

        # Random start indices, one per sequence in the batch.
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # Build x and y by slicing. Cast to int64 because that's what the
        # embedding layer / loss function expect.
        x = torch.stack([
            torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix
        ])

        if self.pin_memory:
            # pin_memory + non_blocking lets the CPU->GPU copy overlap with compute.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

