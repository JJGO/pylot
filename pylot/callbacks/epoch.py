import copy
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Literal, Union

import numpy as np
from pydantic import validate_arguments
from tabulate import tabulate

import torch
import pandas as pd

def PrintLogged(experiment):
    def PrintLoggedCallback(epoch):
        print(f"Logged @ Epoch {epoch}", flush=True)
        df = experiment.metrics.df
        df = df[df.epoch == epoch].drop(columns=["epoch"])
        dfp = pd.pivot(
            pd.melt(df, id_vars="phase", var_name="metric"),
            index="metric",
            columns="phase",
        )
        dfp.columns = [x[1] for x in dfp.columns]
        print(tabulate(dfp, headers="keys"), flush=True)

    return PrintLoggedCallback


def TerminateOnNaN(experiment, monitor="loss"):
    if isinstance(monitor, str):
        monitor = [monitor]

    def TerminateOnNaNCallback(epoch):
        df = experiment.metrics.df
        for metric in monitor:
            if np.isnan(df[metric]).any():
                print(
                    f"Encountered NaN value on {metric} at epoch {epoch}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if np.isinf(df[metric]).any():
                print(
                    f"Encountered Infinity value on {metric} at epoch {epoch}",
                    file=sys.stderr,
                )
                sys.exit(1)

    return TerminateOnNaNCallback


def LogExpr(experiment, **exprs):
    def LogExprCallback(epoch):

        values = {}
        for name, expr in exprs.items():
            values[name] = eval(expr)

        experiment.metricsd["expr"].log({"epoch": epoch, **values})

    return LogExprCallback


class ETA:
    def __init__(self, experiment, n_steps=None, print_freq=1, gamma=0.9):
        if n_steps is None:
            n_steps = experiment.config["train.epochs"] - 1
        self.n_steps = n_steps
        self.timestamps = [None for _ in range(n_steps)]
        self.print_freq = print_freq
        self.gamma = gamma

    def __call__(self, epoch):
        if epoch >= self.n_steps:
            print("Done!")
            return
        self.timestamps[epoch] = time.time()
        if len(self.timestamps) % self.print_freq == 0:
            eta = self._least_squares_fit()
            time_per_epoch = self._time_per_epoch(epoch)
            try:
                eta = datetime.fromtimestamp(eta)
            except ValueError:
                return
            remain = eta - datetime.now()
            remain = self.strfdelta(remain, "{hours:02d}:{minutes:02d}:{seconds:02d}")
            perepoch = self.strfdelta(time_per_epoch, "{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}")
            N = self.n_steps
            print(f"ETA ({epoch}/{N}): {eta:%Y-%m-%d %H:%M:%S} - {remain} remaining - {perepoch} per epoch")

    def _time_per_epoch(self, epoch) -> timedelta:
        if epoch==0:
            return timedelta(0)
        else:
            if self.timestamps[epoch-1] is not None:
                start_time = datetime.fromtimestamp(self.timestamps[epoch-1])
                end_time = datetime.fromtimestamp(self.timestamps[epoch])
                return end_time - start_time
            else:
                for i,t in enumerate(self.timestamps):
                    if t:
                        if i >= (epoch - self.print_freq):
                            break
                current_time = datetime.fromtimestamp(self.timestamps[int(epoch)])
                time_delta = current_time - datetime.fromtimestamp(t)
                return time_delta/max((epoch - i),1)

    def _least_squares_fit(self) -> float:
        # Use weighted least squares with exponentially decaying weights
        # to predict when the iterations will end
        x = np.array([i for i, t in enumerate(self.timestamps) if t])
        y = np.array([t for t in self.timestamps if t])
        A = np.vstack([x, np.ones_like(x)]).T
        w = self.gamma ** np.arange(len(x) - 1, -1, -1)
        Aw = A * np.sqrt(w[:, np.newaxis])
        yw = y * np.sqrt(w)
        m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        eta = m * self.n_steps + c
        return eta

    @staticmethod
    def strfdelta(tdelta, fmt):
        d = {}
        d["hours"], rem = divmod(int(1_000*tdelta.total_seconds()), 1_000*3_600)
        d["minutes"], rem2 = divmod(rem, 1_000*60)
        d["seconds"], d["milliseconds"] = divmod(rem2, 1_000)
        return fmt.format(**d)


class ModelCheckpoint:
    @validate_arguments
    def __init__(
        self,
        experiment,
        monitor: str = "loss",
        mode: Literal["auto", "min", "max"] = "auto",
        phase: str = "val",
        # save_top_k: int = 1,
        save_freq: int = 1,
    ):

        self.phase = phase
        self.experiment = experiment
        self.monitor = monitor
        # self.save_top_k = save_top_k
        self.save_freq = save_freq

        min_patterns = ["*loss*", "*err*"]
        max_patterns = ["*acc*", "*precision*", "*score*"]

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

        self._best_state = None
        self._have_to_save = False

    def __call__(self, epoch):
        # if epoch % self.save_freq != 0:
        #     return
        metrics = self.experiment.metrics.df
        metrics = metrics[metrics.phase == self.phase]
        history = metrics[metrics.epoch < epoch]

        quantity = self.monitor

        tag = f"{self.mode}-{self.phase}-{quantity}"

        prev_best = getattr(history[quantity], self.mode)()
        current_best = getattr(metrics[quantity], self.mode)()

        if prev_best != current_best:
            # ckpt_path = self.experiment.path / "checkpoints"
            # for i in range(self.save_top_k - 1, 0, -1):
            #     src = ckpt_path / f"{tag}_{i}.pt" if i > 1 else ckpt_path / f"{tag}.pt"
            #     if src.exists():
            #         src.rename(ckpt_path / f"{tag}_{i+1}.pt")
            # self.experiment.checkpoint(tag)
            self._best_state = copy.deepcopy(self.experiment.state)
            self._have_to_save = True

        if epoch % self.save_freq == 0 and self._have_to_save:
            if not (self.experiment.path / "checkpoints").exists(): 
                (self.experiment.path / "checkpoints").mkdir(parents=True, exist_ok=True)
            with (self.experiment.path / f"checkpoints/{tag}.pt").open("wb") as f:
                print(f"Saving model with {tag}")
                torch.save(self._best_state, f)
                self._have_to_save = False


def JobProgress(experiment):

    try:
        from syl import JobEnvironment

        job = JobEnvironment().job
        total = experiment.config["train.epochs"]
        if job:
            job["path"] = str(experiment.path)

        def JobProgessCallback(epoch):
            if job:
                job.update_progress(round((epoch + 1) / total, 4))

        return JobProgessCallback

    except ModuleNotFoundError:

        def DummyCallback(epoch):
            pass

        return DummyCallback


def Timestamp(experiment):
    def TimestampCallback(epoch):
        experiment.metricsd["timestamp"].log({"epoch": epoch, "timestamp": time.time()})

    return TimestampCallback
