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
        if self.verbose:
            self._print_elapsed(label, end - start)
        self._save(label, end - start)

    def _print_elapsed(self, label, elapsed):
        print(f"{label} took {elapsed}s")

    def _save(self, label, elapsed):

        self._measurements[label] = elapsed

    @property
    def measurements(self):
        return dict(self._measurements)


class StatsTimer(Timer):

    def __init__(self, verbose=False, cuda=False, skip=0):
        super().__init__(verbose=verbose, cuda=cuda)
        self._skip = defaultdict(lambda: skip)

    def reset(self):
        self._measurements = defaultdict(StatsMeter)

    def _save(self, label, elapsed):
        if self._skip[label] > 0:
            self._skip[label] -= 1
        else:
            self._measurements[label].add(elapsed)

    def _print_elapsed(self, label, elapsed):
        if self._skip[label] > 0:
            print(f"{label} took {elapsed}s (skipped)")
        else:
            print(f"{label} took {elapsed}s")

    def skip(self, label, instances=1):
        self._skip[label] += instances
