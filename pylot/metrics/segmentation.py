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
    y_pred: Tensor, y_true: Tensor, mode: InputMode = "auto", from_logits: bool = False,
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
        y_pred, y_true, mode=mode, from_logits=from_logits, discretize=True,
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
        y_pred, y_true, mode=mode, from_logits=from_logits, discretize=True,
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


def IoU(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Literal["mean", None] = "mean",
    batch_reduction: Literal["mean", None] = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = 0,
    from_logits: bool = False,
) -> Tensor:
    """
    Implemented like jaccard, but with the important distinction that
    classes that don't have label present don't contribute towards the mean
    and background is ignored by default
    """
    if ignore_index is not None and ignore_index != 0:
        raise NotImplementedError

    assert y_true.dtype == torch.int64
    C = y_pred.shape[1]
    flat_true = y_true.reshape(y_true.shape[0], -1)
    mask = torch.stack(
        [torch.bincount(row, minlength=C) for row in torch.unique(flat_true, dim=1)]
    )

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits, discretize=True,
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    cardinalities = (y_pred == 1.0).sum(dim=-1) + (y_true == 1.0).sum(dim=-1)
    union = cardinalities - intersection

    score = (intersection) / (union + smooth).clamp_min(eps)
    score = score * mask

    if ignore_index == 0:
        score = score[:, 1:]
        mask = mask[:, 1:]

    if reduction == "mean":
        score = score.sum(dim=1) / mask.sum(dim=1)

    if batch_reduction == "mean":
        assert reduction == "mean", "Can't have reduction=None and batch_reduction=mean"
        score = score.nanmean()

    return score




@validate_arguments(config=dict(arbitrary_types_allowed=True))
def mIoU(
    y_pred: Tensor,
    y_true: Tensor,
    ignore_index: Optional[int] = 0,
    reduction: Optional[Literal["mean"]] = "mean",
    eps: float = 1e-15,
):
    from torchmetrics.functional.classification import confusion_matrix
    if ignore_index is not None and ignore_index != 0:
        raise NotImplementedError

    cm = confusion_matrix(
        y_pred, y_true, task="multiclass", num_classes=y_pred.shape[1]
    )
    iou = cm.diag() / (cm.sum(axis=0) + cm.sum(axis=1) - cm.diag())
    if ignore_index == 0:
        iou = iou[1:]
    if reduction == "mean":
        iou = torch.nanmean(iou)
    return iou


# @validate_arguments(config=dict(arbitrary_types_allowed=True))
# def mIoU(
#     y_pred: Tensor,
#     y_true: Tensor,
#     mode: InputMode = "auto",
#     batch_reduction: Literal[None, "mean"] = "mean",
#     from_logits: bool = True,
#     ignore_index: Optional[int] = 0,
# ):
#     """
#     WARNING!

#     This function, despite being correct, runs extremely slowly on CUDA.
#     """
#     y_pred, y_true = _inputs_as_longlabels(
#         y_pred, y_true, mode=mode, from_logits=from_logits, discretize=True
#     )
#     C = y_pred.shape[1]

#     if ignore_index is not None and ignore_index != 0:
#         raise NotImplementedError

#     vals = []
#     for pred, target in zip(y_pred, y_true):
#         # treat every element in the batch independently
#         intersection = torch.where(pred == target, target, torch.zeros_like(target))
#         area_intersection = torch.bincount(intersection.flatten(), minlength=C)
#         area_target = torch.bincount(target.flatten(), minlength=C)
#         area_pred = torch.bincount(pred.flatten(), minlength=C)
#         area_union = area_target + area_pred - area_intersection
#         iou = area_intersection / area_union

#         if ignore_index == 0:
#             iou = iou[1:]
#         # nan-mean ignores 0/0 cases, i.e. no label predicted, nor expected
#         miou = torch.nanmean(iou)
#         vals.append(miou)

#     mious = torch.stack(vals)
#     if batch_reduction is None:
#         return mious
#     elif batch_reduction == "mean":
#         return torch.mean(mious)
#     raise ValueError(f"Unsupported batch reduction {batch_reduction}")
