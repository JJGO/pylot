from typing import Optional, Union, List, Literal

import torch
from torch import Tensor
import torch.nn.functional as F
from pydantic import validate_arguments


from .util import (
    _metric_reduction,
    _inputs_as_onehot,
    _inputs_as_longlabels,
    InputMode,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_longlabels(
        y_pred, y_true, mode, from_logits=from_logits, discretize=True
    )
    correct = y_pred == y_true
    return correct.float().mean()


def pixel_mse(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    # per_channel: bool = False,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    # if per_channel:
    # NOTE: Each channel is weighted equally because of the mean reduction
    correct = (y_pred - y_true).square().mean(dim=-1)

    return _metric_reduction(
        correct,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )
    # else:
    #     return F.mse_loss(y_pred, y_true, reduction=reduction)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=-1)

    if square_denom:
        cardinalities = y_pred.square().sum(dim=-1) + y_true.square().sum(dim=-1)
    else:
        cardinalities = y_pred.sum(dim=-1) + y_true.sum(dim=-1)

    score = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=-1)

    if square_denom:
        cardinalities = y_pred.square().sum(dim=-1) + y_true.square().sum(dim=-1)
    else:
        cardinalities = y_pred.sum(dim=-1) + y_true.sum(dim=-1)

    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
        discretize=True,
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    cardinalities = (y_pred == 1.0).sum(dim=-1) + (y_true == 1.0).sum(dim=-1)

    score = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)

    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


def jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
        discretize=True,
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    cardinalities = (y_pred == 1.0).sum(dim=-1) + (y_true == 1.0).sum(dim=-1)
    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )
