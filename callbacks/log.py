from datetime import datetime
import time

from torch import nn
import numpy as np


def LogParameters(experiment, parameters):
    def LogParametersCallback(epoch):
        param_dict = {}
        for parameter in parameters:
            param = getattr(experiment.model, parameter)
            if isinstance(param, (nn.ParameterList, list)):
                for i, p in enumerate(param):
                    param_dict[f"{parameter}_{i}"] = p.item()
            else:
                param_dict[parameter] = param.item()
        experiment.log(**param_dict)

    return LogParametersCallback


def TqdmParameters(experiment, parameters):
    def TqdmParametersCallback(train, epoch, iteration, postfix):
        param_dict = {}
        for parameter in parameters:
            param = getattr(experiment.model, parameter)
            if isinstance(param, (list, nn.ParameterList)):
                for i, p in enumerate(param):
                    param_dict[f"{parameter}_{i}"] = p.item()
            else:
                param_dict[parameter] = param.item()
        postfix.update(param_dict)

    return TqdmParametersCallback


def PrintLogged(experiment):
    def PrintLoggedCallback(epoch):
        print(f"Logged @ Epoch {epoch}", flush=True)
        csv = experiment.csvlogger
        for c, v in zip(csv.columns, csv.values[-1]):
            if isinstance(v, (int, float)):
                print(f"{c:<20s}: {v:n}", flush=True)
            else:
                print(f"{c:<20s}: {v}", flush=True)

    return PrintLoggedCallback


class ETA:
    def __init__(self, experiment, n_steps, print_freq=1, gamma=0.9):
        self.n_steps = n_steps
        # self.counter = 0
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
            eta = datetime.fromtimestamp(eta)
            remain = eta - datetime.now()
            remain = self.strfdelta(remain, "{hours:02d}:{minutes:02d}:{seconds:02d}")
            N = self.n_steps
            print(f"ETA ({epoch}/{N}): {eta:%Y-%m-%d %H:%M:%S} - {remain} remaining")

    def _least_squares_fit(self):
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
        d["hours"], rem = divmod(int(tdelta.total_seconds()), 3600)
        d["minutes"], d["seconds"] = divmod(rem, 60)
        return fmt.format(**d)
