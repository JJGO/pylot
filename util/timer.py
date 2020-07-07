from collections import defaultdict
from contextlib import contextmanager
import time

import torch

from .meter import StatsMeter

UNIT_FACTORS = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "m": 1 / 60, "h": 1 / 3600}


class Timer:
    def __init__(self, verbose=False, unit="s", enabled=True):
        self.verbose = verbose
        self.reset()
        self.unit = unit
        self._factor = UNIT_FACTORS[unit]
        self.enabled = enabled

    def reset(self):
        self._measurements = {}

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
        if self.verbose:
            print(f"{label} took {elapsed}{self.unit}")
        self._measurements[label] = elapsed * self._factor

    @property
    def measurements(self):
        return dict(self._measurements)


class StatsTimer(Timer):
    # TODO: Add n_samples param so once n_samples is collected for a
    # key then the timing is not collected anymore
    def __init__(self, verbose=False, unit="s", skip=0, enabled=True):
        super().__init__(verbose=verbose, unit=unit, enabled=enabled)
        self._skip = defaultdict(lambda: skip)

    def reset(self):
        self._measurements = defaultdict(StatsMeter)

    def _save(self, label, elapsed):
        if self._skip[label] > 0:
            self._skip[label] -= 1
            print(
                f"{label} took {elapsed}{self.unit} (skipped)"
            ) if self.verbose else None
        else:
            self._measurements[label].add(elapsed * self._factor)
            print(f"{label} took {elapsed}{self.unit}") if self.verbose else None

    def skip(self, label, instances=1):
        self._skip[label] += instances


class CUDATimer(StatsTimer):
    def __init__(self, verbose=False, unit="s", skip=0, enabled=True):
        assert torch.cuda.is_available(), "CUDA not available"
        super().__init__(verbose=verbose, unit=unit, skip=skip, enabled=enabled)
        self._factor /= 1e3

    @contextmanager
    def __call__(self, label=""):
        if self.enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            yield
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            self._save(label, start.elapsed_time(end))
        else:
            yield
