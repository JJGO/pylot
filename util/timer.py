from collections import defaultdict
from contextlib import contextmanager
import functools
import time

import torch

from .meter import StatsMeter

UNIT_FACTORS = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "m": 1 / 60, "h": 1 / 3600}


class Timer:
    def __init__(self, verbose=False, unit="s"):
        self.verbose = verbose
        self.reset()
        self.unit = unit
        self._factor = UNIT_FACTORS[unit]
        self.enabled = True

    def reset(self):
        self._measurements.clear()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    @contextmanager
    def __call__(self, label=""):
        if self.enabled:
            start = time.time()
            yield
            end = time.time()
            self._save(label, end - start)
        else:
            yield

    def _save(self, label, elapsed):
        self._print(f"{label} took {elapsed}{self.unit}")
        self._measurements[label] = elapsed * self._factor

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @property
    def measurements(self):
        return dict(self._measurements)

    def wrap(self, func, label=None):
        if label is None:
            label = func.__name__

        @functools.wraps(func)
        def timed_func(*args, **kwargs):
            with self(label):
                return func(*args, **kwargs)

        return timed_func


class StatsTimer(Timer):
    def __init__(self, verbose=False, unit="s", skip=0, n_samples=None):
        self._skip = defaultdict(lambda: skip)
        super().__init__(verbose=verbose, unit=unit)
        if n_samples is None:
            n_samples = float("inf")
        self.n_samples = n_samples

    def reset(self):
        self._measurements = defaultdict(StatsMeter)
        self._skip.clear()

    def _save(self, label, elapsed):
        if self._skip[label] > 0:
            self._skip[label] -= 1
            self._print(f"{label} took {elapsed}{self.unit} (skipped)")
        else:
            self._measurements[label].add(elapsed * self._factor)
            self._print(f"{label} took {elapsed}{self.unit}")

    def skip(self, label, instances=1):
        self._skip[label] += instances

    @contextmanager
    def __call__(self, label=""):
        if self.enabled and self._measurements[label].n < self.n_samples:
            start = time.time()
            yield
            end = time.time()
            self._save(label, end - start)
        else:
            yield


class CUDATimer(StatsTimer):
    def __init__(self, verbose=False, unit="s", skip=0, n_samples=None):
        assert torch.cuda.is_available(), "CUDA not available"
        super().__init__(
            verbose=verbose, unit=unit, skip=skip, n_samples=n_samples,
        )
        self._factor /= 1e3  # CUDA Evants are measured in ms

    @contextmanager
    def __call__(self, label=""):
        if self.enabled and self._measurements[label].n < self.n_samples:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            start.record()
            yield
            end.record()
            # Waits for everything to finish running
            # Sync after record is right despite being counterintuitive
            torch.cuda.synchronize()
            self._save(label, start.elapsed_time(end))
        else:
            yield
