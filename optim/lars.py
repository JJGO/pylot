"""
References:
    - https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/optimizers/lars_scheduling.py
    - https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
"""

import torch
from torch.optim import SGD

from .wrapper import OptimWrapper
from ..util import delegates, separate_kwargs

# from torchlars._adaptive_lr import compute_adaptive_lr # Impossible to build extensions


__all__ = ["LARS"]


class LARS(OptimWrapper):
    """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
    :class:`~torch.optim.Optimizer` wrapper.

    __ : https://arxiv.org/abs/1708.03888

    Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
    you want to the same performance obtained with small-batch training when
    you use large-batch training, LARS will be helpful::

    Args:
        optimizer (Optimizer):
            optimizer to wrap
        eps (float, optional):
            epsilon to help with numerical stability while calculating the
            adaptive learning rate
        trust_coef (float, optional):
            trust coefficient for calculating the adaptive learning rate

    Example::
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = LARS(optimizer=base_optimizer)

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

    """

    def __init__(self, optimizer, trust_coef=0.02, clip=True, eps=1e-8):
        if eps < 0.0:
            raise ValueError("invalid epsilon value: , %f" % eps)
        if trust_coef < 0.0:
            raise ValueError("invalid trust coefficient: %f" % trust_coef)

        self.optim = optimizer
        self.eps = eps
        self.trust_coef = trust_coef
        self.clip = clip

    def __getstate__(self):
        self.optim.__get
        lars_dict = {}
        lars_dict["trust_coef"] = self.trust_coef
        lars_dict["clip"] = self.clip
        lars_dict["eps"] = self.eps
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state
        self.trust_coef = lars_dict["trust_coef"]
        self.clip = lars_dict["clip"]
        self.eps = lars_dict["eps"]

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            for p in group["params"]:
                if p.grad is not None:
                    self.update_p(p, group, weight_decay)

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            divisor = g_norm + p_norm * weight_decay + self.eps
            adaptive_lr = (self.trust_coef * p_norm) / divisor

            # clip lr
            if self.clip:
                adaptive_lr = min(adaptive_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= adaptive_lr




class SGDLARS(LARS):
    @delegates(to=LARS.__init__)
    @delegates(to=SGD.__init__, keep=True, but=["eps", "trust_coef"])
    def __init__(self, params, **kwargs):
        sgd_kwargs, lars_kwargs = separate_kwargs(kwargs, SGD.__init__)
        optim = SGD(params, **sgd_kwargs)
        super().__init__(optim, **lars_kwargs)
