from collections import defaultdict
from contextlib import contextmanager
import time

import torch

from .meter import StatsMeter

UNIT_FACTORS = {
    's': 1,
    'ms': 1e3,
    'us': 1e6,
    'ns': 1e9,
    'm': 1 / 60,
    'h': 1 / 3600
}


class Timer:

    def __init__(self, verbose=False, unit='s'):
        self.verbose = verbose
        self.reset()
        self.unit = unit
        self._factor = UNIT_FACTORS[unit]

    def reset(self):
        self._measurements = {}

    @contextmanager
    def __call__(self, label=""):

        start = time.time()
        yield
        end = time.time()
        self._save(label, end - start)

    def _save(self, label, elapsed):
        if self.verbose:
            print(f"{label} took {elapsed}{self.unit}")
        self._measurements[label] = elapsed * self._factor

    @property
    def measurements(self):
        return dict(self._measurements)


class StatsTimer(Timer):

    def __init__(self, verbose=False, unit='s', skip=0):
        super().__init__(verbose=verbose, unit=unit)
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
