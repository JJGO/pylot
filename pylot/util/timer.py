from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import time

import torch
import torch.cuda
import pandas as pd

from .meter import StatsMeter

UNIT_FACTORS = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "m": 1 / 60, "h": 1 / 3600}


class Timer:
    def __init__(self, verbose=False, unit="s"):
        self.verbose = verbose
        self._measurements = {}
        self.unit = unit
        self._factor = UNIT_FACTORS[unit]
        self.enabled = True

    def reset(self):
        self._measurements.clear()
        return self

    def enable(self):
        self.enabled = True
        return self

    def disable(self):
        self.enabled = False
        return self

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
            self._print(f"{label} took {elapsed:.2g}{self.unit}")
        self._measurements[label] = elapsed * self._factor

    def _print(self, *args, **kwargs):
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

    def __getitem__(self, label):
        return self._measurements[label]


class HistoryTimer(Timer):
    def __init__(self, verbose=False, unit="s", skip=0, max_history=0):
        super().__init__(verbose=verbose, unit=unit)
        self._measurements = defaultdict(deque)
        self._skip = defaultdict(lambda: skip)
        self.max_history = max_history

    def _save(self, label, elapsed):
        if self._skip[label] > 0:
            self._skip[label] -= 1
            if self.verbose:
                self._print(f"{label} took {elapsed}{self.unit} (skipped)")
        else:
            self._measurements[label].append(elapsed * self._factor)
            if (
                self.max_history > 0
                and len(self._measurements[label]) > self.max_history
            ):
                self._measurements[label].popleft()
            if self.verbose:
                self._print(f"{label} took {elapsed}{self.unit}")

    def reset(self):
        self._measurements = defaultdict(deque)
        self._skip.clear()

    # def wrap_generator(self, generator, label=None):
    #     if label is None:
    #         label = "_iter"

    #     it = iter(generator)

    #     def timed_generator():
    #         while
    #         # while True:
    #         #         item = next(it)
    #         #     with self(label):
    #         #     yield item

    #     return timed_generator()


class StatsTimer(Timer):
    def __init__(self, verbose=False, unit="s", skip=0, n_samples=None):
        self._skip = defaultdict(lambda: skip)
        super().__init__(verbose=verbose, unit=unit)
        if n_samples is None:
            n_samples = float("inf")
        self.n_samples = n_samples
        self._measurements = defaultdict(StatsMeter)

    def reset(self):
        self._measurements = defaultdict(StatsMeter)
        self._skip.clear()

    def _save(self, label, elapsed):
        if self._skip[label] > 0:
            self._skip[label] -= 1
            if self.verbose:
                self._print(f"{label} took {elapsed}{self.unit} (skipped)")
        else:
            self._measurements[label].add(elapsed * self._factor)
            if self.verbose:
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

    def measurements_df(self):
        rows = []
        for label, stats in self._measurements.items():
            rows.append(
                {"label": label, "mean": stats.mean, "std": stats.std, "n": stats.n}
            )
        return pd.DataFrame.from_records(rows)


class StatsCUDATimer(StatsTimer):
    def __init__(self, verbose=False, unit="s", skip=0, n_samples=None):
        assert torch.cuda.is_available(), "CUDA not available"
        super().__init__(
            verbose=verbose,
            unit=unit,
            skip=skip,
            n_samples=n_samples,
        )
        self._factor /= 1e3  # CUDA Events are measured in ms

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


class HistoryCUDATimer(HistoryTimer):
    def __init__(self, verbose=False, unit="s", skip=0, max_history=0):
        assert torch.cuda.is_available(), "CUDA not available"
        super().__init__(verbose=verbose, unit=unit)
        self._measurements = defaultdict(deque)
        self._skip = defaultdict(lambda: skip)
        self.max_history = max_history
        self._factor /= 1e3  # CUDA Events are measured in ms

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


class CUDATimer(Timer):
    def __init__(self, verbose=False, unit="s"):
        assert torch.cuda.is_available(), "CUDA not available"
        super().__init__(verbose=verbose, unit=unit)
        self._factor /= 1e3  # CUDA Evants are measured in ms

    @contextmanager
    def __call__(self, label=""):
        if self.enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            self._save(label, start.elapsed_time(end))
        else:
            yield
