import torch
from torch import nn

from ..metrics.dice import soft_dice, hard_dice


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1e-10, agg="mean", reduction="mean", background=None):
        super().__init__()
        self.smooth = smooth
        self.agg = agg
        self.reduction = reduction
        # background param specifies which channel is background and
        # should be ignored in loss computation
        self.background = background

    def forward(self, input, target):

        weights = None
        if self.background:
            weights = torch.ones(size=(input.shape[1]), device=input.device)
            weights[self.background] = 0

        # With mean aggregation, 1-soft_dice \in [0,1] where 0 is best
        loss = 1 - self.dice(input, target, smooth=self.smooth, agg=self.agg, weights=weights)

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean(dim=0)
        elif self.reduction == "sum":
            return loss.sum(dim=0)

    def dice(self, *args, **kwargs):
        return soft_dice(*args, **kwargs)


class HardDiceLoss(SoftDiceLoss):

    def dice(self, *args, **kwargs):
        return hard_dice(*args, **kwargs)
