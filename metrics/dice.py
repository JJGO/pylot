import torch
import torch.nn.functional as F

from .util import batch_channel_flatten
from ..util.meta import delegates


def soft_dice(input, target, smooth=1e-10, agg="mean", weights=None):

    assert (
        input.shape == target.shape
    ), f"Input {input.shape} and Target {target.shape} must be same shape in Soft Dice"
    if weights:
        assert (
            weights.shape[0] == input.shape[1]
        ), "Weights size must correspond to number of channels"

    # Flattens by default assuming (batch_size, n_channels, *spatial_dims)
    input = batch_channel_flatten(input)
    target = batch_channel_flatten(target)

    intersection = (input * target).sum(dim=-1)
    pseudounion = input.square().sum(dim=-1) + target.square().sum(dim=-1)

    dice = (2 * intersection + smooth) / (pseudounion + smooth)

    if weights:
        dice *= weights[None, :]

    # How to aggregate the channel dimension
    if agg is None:
        return dice
    elif agg == "mean":
        return dice.mean(dim=1)
    elif agg == "sum":
        return dice.sum(dim=1)
    else:
        raise ValueError("Aggregation mode must be one of None|sum|mean")


def hard_max(x):
    N = len(x.shape)
    order = (0, N - 1, *[i for i in range(1, N - 1)])
    return F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).permute(order)


@delegates(to=soft_dice)
def hard_dice(input, target, **kwargs):
    input = hard_max(input)
    target = hard_max(target)
    return soft_dice(input, target, **kwargs)
