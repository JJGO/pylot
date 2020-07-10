from datetime import datetime
import time

from torch import nn
import numpy as np


def LogParameters(experiment, parameters):
    def LogParametersCallback(experiment, epoch):
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
    def TqdmParametersCallback(experiment, postfix):
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
    def PrintLoggedCallback(experiment, epoch):
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
        self.timestamps = [time.time()]
        self.print_freq = print_freq
        self.gamma = gamma

    def __call__(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) % self.print_freq == 0:
            eta = self._least_squares_fit()
            eta = datetime.fromtimestamp(eta)
            n = len(self.timestamps) - 1
            print(f"ETA ({n}/{self.n_steps}): {eta:%Y-%m-%d %H:%M:%S}")

    def _least_squares_fit(self):
        # Use weighted least squares with exponentially decaying weights
        # to predict when the iterations will end
        n = len(self.timestamps)
        x = np.arange(0, n)
        y = np.array(self.timestamps)
        A = np.vstack([x, np.ones_like(x)]).T
        w = self.gamma ** np.arange(n - 1, -1, -1)
        Aw = A * np.sqrt(w[:, np.newaxis])
        yw = y * np.sqrt(w)
        m, c = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        eta = m * self.n_steps + c
        return eta
