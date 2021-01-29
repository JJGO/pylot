import torch
from torch.optim import Optimizer


class OptimWrapper(Optimizer):

    # Mixin class that defines convenient functions for writing Optimizer Wrappers

    def __init__(self, optim):
        self.optim = optim

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @torch.no_grad()
    def step(self, closure=None):
        self.optim.step(closure=closure)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.optim)
