from collections import defaultdict
from contextlib import contextmanager
import time

import torch

from .meter import StatsMeter


class Timer:

    def __init__(self, verbose=False, cuda=False):
        self.verbose = verbose
        self.reset()
        self.cuda = cuda
        if self.cuda:
            assert torch.cuda.is_available()

    def reset(self):
        self._measurements = {}

    @contextmanager
    def __call__(self, label):

        if self.cuda:
            torch.cuda.synchronize()
        start = time.time()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        end = time.time()
        self._save(label, end-start)

    def _save(self, label, elapsed):

        self._measurements[label] = elapsed
        if self.verbose:
            print(f"{label} took {elapsed:n}s")

    @property
    def measurements(self):
        return self._measurements.copy()


class StatsTimer(Timer):

    def reset(self):
        self._measurements = defaultdict(StatsMeter)

    def _save(self, label, elapsed):
        self._measurements[label].add(elapsed)
