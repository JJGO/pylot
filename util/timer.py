from contextlib import contextmanager
import time

import torch


class Timer:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self._measurements = {}

    @contextmanager
    def __call__(self, label):

        start = time.time()
        yield
        end = time.time()
        self._save(label, end-start)

    def _save(self, label, elapsed):

        self._measurements[label] = elapsed
        if self.verbose:
            print(f"{label} took {elapsed:n}s")

    @property
    def measurements(self):
        return self._measurements.copy()


class CUDATimer(Timer):

    def __init__(self, verbose=False):
        assert torch.cuda.is_available()
        super().__init__(verbose)

    @contextmanager
    def __call__(self, label):
        torch.cuda.synchronize()
        start = time.time()
        yield
        torch.cuda.synchronize()
        end = time.time()
        self._save(label, end-start)
