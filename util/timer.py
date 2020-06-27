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
    def __call__(self, label=""):

        if self.cuda:
            torch.cuda.synchronize()
        start = time.time()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        end = time.time()
        self._save(label, end-start)
        if self.verbose:
            print(f"{label} took {end-start:n}s")

    def _save(self, label, elapsed):

        self._measurements[label] = elapsed

    @property
    def measurements(self):
        return dict(self._measurements)


class StatsTimer(Timer):

    def reset(self):
        self._measurements = defaultdict(StatsMeter)

    def _save(self, label, elapsed):
        self._measurements[label].add(elapsed)
