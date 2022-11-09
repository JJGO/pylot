import sys
from typing import Literal
from fnmatch import fnmatch

import numpy as np
from pydantic import validate_arguments


class EarlyStopping:
    @validate_arguments
    def __init__(
        self,
        experiment,
        monitor: str = "loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["auto", "min", "max"] = "auto",
        check_finite: bool = True,
        phase: str = "val",
    ):

        assert min_delta >= 0.0

        self.experiment = experiment
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.phase = phase
        self.check_finite = check_finite

        self._set_mode(mode)

    def _set_mode(self, mode):
        min_patterns = ["*loss*", "*err*"]
        max_patterns = ["*acc*", "*precision*", "*score*", "*roc"]

        if mode == "auto":
            self.mode = None
            m = self.monitor
            if any(fnmatch(m, pat) for pat in min_patterns):
                self.mode = "min"
            elif any(fnmatch(m, pat) for pat in max_patterns):
                self.mode = "max"
            else:
                raise ValueError(f"Can't infer mode for metric {m}")
        else:
            self.mode = mode

    def __call__(self, epoch):
        metrics = self.experiment.metrics.df
        metrics = metrics[metrics.phase == self.phase]

        if self.check_finite and not np.isfinite(metrics[self.monitor]).all():
            print(f"Encountered non-finite value for {self.phase} {self.monitor}", file=sys.stderr)
            sys.exit(1)
 
        quantity = self.monitor

        last_epoch = metrics.epoch.max()

        previous = metrics[metrics.epoch < last_epoch - self.patience]
        recent = metrics[metrics.epoch >= last_epoch - self.patience]

        fn = {'max': np.max, 'min': np.min}[self.mode]
        cmp = {'max': np.greater, 'min': np.less}[self.mode]
        # Non-recent is better than recent
        if cmp( fn(previous[quantity]), fn(recent[quantity])):
            print(f"{quantity} has not improved for {self.patience} epochs", file=sys.stderr)
            sys.exit(1)

        if abs( np.min(recent[quantity] - np.max(recent[quantity]) ) < self.min_delta:
            print(f"{quantity} has improved less than {self.min_delta} in the last {self.patience} epochs", file=sys.stderr)
            sys.exit(1)



