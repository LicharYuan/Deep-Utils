import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import itertools
import uuid
import random
import numpy as np

class InfiniteSampler(Sampler):
    """build infinite sampler when training"""
    def __init__(
        self,
        size,
        shuffle=True,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size