import inspect
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, losses, weights=None, return_all=True):
        super().__init__()
        if weights is None:
            weights = [1 for _ in losses]
        assert len(weights) == len(losses)
        self.weights = weights
        self.losses = losses
        self.return_all = return_all

        def get_name(loss):
            if inspect.isfunction(loss):
                return loss.__name__
            if inspect.ismethod(loss):
                return loss.__class__.__name__

        self.names = [get_name(loss_func) for loss_func in losses]

    def forward(self, pred, target):
        if not self.return_all:
            return sum(w * fn(pred, target) for w, fn in zip(self.weights, self.losses))
        losses = [(name, fn(pred, target)) for name, fn in zip(self.names, self.losses)]
        losses.append(("all", sum(w * loss for w, loss in zip(self.weights, losses))))
        return dict(losses)
