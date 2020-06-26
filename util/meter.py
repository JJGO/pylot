from abc import abstractmethod
import numpy as np
# from collections import defaultdict

from .csvlogger import CSVLogger


class Meter:

    def __init__(self, iterable=None):
        if iterable is not None:
            self.addN(iterable)

    @abstractmethod
    def add(self, datum):
        pass

    def addN(self, iterable):
        for datum in iterable:
            self.add(datum)


class StatsMeter(Meter):
    """
    Auxiliary classs to keep track of online stats including:
        - mean
        - std / variance
    Uses Welford's algorithm to compute sample mean and sample variance incrementally.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    """

    def __init__(self, iterable=None):
        """Online Mean and Variance from single samples

        Running stats,
        This is compatible with np.ndarray objects and as long as the

        Keyword Arguments:
            iterable {[iterable} -- Values to initialize (default: {None})
        """
        self.n = 0
        self.mean = 0.0
        self.S = 0.0
        super().__init__(iterable)

    def add(self, datum):
        """Add a single datum

        Internals are updated using Welford's method

        Arguments:
            datum  -- Numerical object
        """
        self.n += 1
        delta = datum - self.mean
        # Mk = Mk-1+ (xk – Mk-1)/k
        self.mean += delta / self.n
        # Sk = Sk-1 + (xk – Mk-1)*(xk – Mk).
        self.S += delta * (datum - self.mean)

    def addN(self, iterable, batch=False):
        """Add N data to the stats

        Arguments:
            iterable {[type]} -- [description]

        Keyword Arguments:
            batch {bool} -- If true, then the mean and std are computed over
            the new array using numpy and then that updates the current stats
        """
        if batch:
            add = self + StatsMeter.from_values(len(iterable), np.mean(iterable), np.std(iterable))
            self.n, self.mean, self.S = add.n, add.mean, add.S
        else:
            super().addN(iterable)

    def pop(self, datum):
        if self.n == 0:
            raise ValueError("Stats must be non empty")

        self.n -= 1
        delta = datum - self.mean
        # Mk-1 = Mk - (xk - Mk) / (k - 1)
        self.mean -= delta / self.n
        # Sk-1 = Sk - (xk – Mk-1) * (xk – Mk)
        self.S -= (datum - self.mean) * delta

    def popN(self, iterable, batch=False):
        if batch:
            raise NotImplementedError
        else:
            for datum in iterable:
                self.pop(datum)

    @property
    def variance(self):
        # For 2 ≤ k ≤ n, the kth estimate of the variance is s2 = Sk/(k – 1).
        return self.S / self.n

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def flatmean(self):
        # for datapoints which are arrays
        return np.mean(self.mean)

    @property
    def flatvariance(self):
        # for datapoints which are arrays
        return np.mean(self.variance+self.mean**2) - self.flatmean**2

    @property
    def flatstd(self):
        return np.sqrt(self.flatvariance)

    @staticmethod
    def from_values(n, mean, std):
        stats = StatsMeter()
        stats.n = n
        stats.mean = mean
        stats.S = std**2 * n
        return stats

    @staticmethod
    def from_raw_values(n, mean, S):
        stats = StatsMeter()
        stats.n = n
        stats.mean = mean
        stats.S = S
        return stats

    def __str__(self):
        return f"n={self.n}  mean={self.mean}  std={self.std}"

    def __repr__(self):
        return "StatsMeter.from_values(" + \
               f"n={self.n}, mean={self.mean}, " + \
               f"std={self.std})"

    def __add__(self, other):
        """Adding can be done with int|float or other Online Stats

        For other int|float, it is added to all previous values

        Arguments:
            other {[type]} -- [description]

        Returns:
            StatsMeter -- New instance with the sum.

        Raises:
            TypeError -- If the type is different from int|float|OnlineStas
        """
        if isinstance(other, StatsMeter):
            # Add the means, variances and n_samples of two objects
            n1, n2 = self.n, other.n
            mu1, mu2 = self.mean, other.mean
            S1, S2 = self.S, other.S
            # New stats
            n = n1 + n2
            mu = n1/n * mu1 + n2/n * mu2
            S = (S1 + n1 * mu1*mu1) + (S2 + n2 * mu2*mu2) - n * mu*mu
            return StatsMeter.from_raw_values(n, mu, S)
        if isinstance(other, (int, float)):
            # Add a fixed amount to all values. Only changes the mean
            return StatsMeter.from_raw_values(self.n, self.mean+other, self.S)
        else:
            raise TypeError("Can only add other groups or numbers")

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, k):
        # Multiply all values seen by some constant
        return StatsMeter.from_raw_values(self.n, self.mean*k, self.S*k**2)

    def asdict(self):
        return {'mean': self.mean, 'std': self.std}  #, 'n': self.n}


class MaxMinMeter(Meter):

    def __init__(self, iterable=None):
        self. n = 0
        self.max_ = float('-inf')
        self.min_ = float('inf')
        super().__init__(iterable)

    def add(self, datum):
        self.n += 1
        self.max_ = max(datum, self.max_)
        self.min_ = min(datum, self.min_)

    @property
    def max(self):
        return self.max_

    @property
    def min(self):
        return self.min_

    def asdict(self):
        return {'min': self.min, 'max': self.max}  #, 'n': self.n}


class UnionMeter(Meter):

    def __init__(self, meters, iterable=None):
        assert all(isinstance(m, Meter) for m in meters)
        self.meters = meters
        super().__init__(iterable)

    def add(self, datum):
        for m in self.meters:
            m.add(datum)

    def asdict(self):
        d = {}
        for meter in self.meters:
            d.update(meter.asdict())
        return d

    def __getattr__(self, attr):

        for meter in self.meters:
            if hasattr(meter, attr):
                return getattr(meter, attr)
        else:
            raise AttributeError(f"No meter has attribute {attr}")

    @staticmethod
    def union(*meter_types):
        assert all(issubclass(m, Meter) for m in meter_types)

        def constructor():
            return UnionMeter([mt() for mt in meter_types])

        return constructor


# class StatsMeterMap:

#     def __init__(self, *keys):
#         self.stats = defaultdict(StatsMeter)
#         if keys is not None:
#             self.register(*keys)

#     def register(self, *keys):
#         for k in keys:
#             self.stats[k]

#     def __iter__(self):
#         return self.stats

#     def __str__(self):
#         s = "Stats"
#         max_len = max(len(k) for k in self.stats)
#         for k in self:
#             s += f'  {k:>{max_len}s}:  {str(self.stats[k])}'


class MeterCSVLogger(CSVLogger):

    def set(self, *args, **kwargs):

        def _flatten_stats(mapping):
            new = {}
            for k, v in mapping.items():
                if isinstance(v, Meter):
                    for param, v2 in v.asdict().items():
                        new[f"{k}_{param}"] = v2
                else:
                    new[k] = v

            return new

        args = [_flatten_stats(mapping) for mapping in args]
        kwargs = _flatten_stats(kwargs)

        super().set(*args, **kwargs)
