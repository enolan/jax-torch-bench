import numpy as np
import numpy.typing as npt
import random
from typing import Any


class Enwik9Loader:
    """Iterator that returns shuffled slices of Enwik9"""

    def __init__(self, batch_size: int, seq_len: int):
        self.arr = np.fromfile("/home/enolan/junk/enwik9", dtype=np.uint8)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        # Make slice boundaries randomized across epochs
        offset = random.randint(0, self.seq_len - 1)
        offset_len = self.arr.size - offset
        seqs = offset_len // self.seq_len
        slices = np.array(
            [
                self.arr[start : start + self.seq_len]
                for start in range(offset, offset + seqs * self.seq_len, self.seq_len)
            ]
        )
        np.random.default_rng().shuffle(slices)
        short_batch = len(slices) % self.batch_size
        batches = [
            slices[start : start + self.batch_size]
            for start in range(0, len(slices) - short_batch, self.batch_size)
        ]
        return iter(batches)
