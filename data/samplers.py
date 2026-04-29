import itertools

import numpy as np
from torch.utils.data.sampler import Sampler


def _group_indices(iterable, batch_size):
    args = [iter(iterable)] * batch_size
    return zip(*args)


def _make_random_state(seed, epoch):
    if seed is None:
        return np.random.RandomState(None)
    return np.random.RandomState(seed + epoch)


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, seed=0):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = _make_random_state(self.seed, self.epoch)
        primary_iter = rng.permutation(self.primary_indices)
        secondary_iter = itertools.chain.from_iterable(
            rng.permutation(self.secondary_indices)
            for _ in itertools.repeat(None)
        )
        return (
            list(p) + list(s)
            for p, s in zip(
                _group_indices(primary_iter, self.primary_batch_size),
                _group_indices(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class OneStreamBatchSampler(Sampler):
    def __init__(self, labeled_indices, batch_size, seed=0):
        self.labeled_indices = labeled_indices
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = _make_random_state(self.seed, self.epoch)
        indices = rng.permutation(self.labeled_indices)
        return (list(batch) for batch in _group_indices(indices, self.batch_size))

    def __len__(self):
        return len(self.labeled_indices) // self.batch_size
