from typing import Tuple, Optional, Union, List

import torch
from torch import Tensor
import torch.nn.functional as F

from .util import _metric_reduction, _inputs_as_onehot, _inputs_as_longlabels
from ..util import delegates


# def as_onehot(y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
#     if isinstance(y_true, (torch.LongTensor, torch.cuda.LongTensor)):
#         num_classes = y_pred.shape[1]
#         y_true = F.onehot(y_true, num_classes).float()


# # def as_labels(y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor]:
# #     y_pred = torch.argmax(y_pred, dim=1).long()
# #     if isinstance(y_true, (torch.LongTensor, torch.cuda.LongTensor)):
# #         y_true = torch.argmax(y_true, dim=1).long()
# #     return y_pred, y_true


def pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
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
    mode: str = "auto",
    per_channel: bool = False,
    from_logits: bool = False,
    reduction: str = "mean",
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    if per_channel:
        # NOTE: Each channel is weighted equally because of the mean reduction
        correct = (y_pred - y_true).square().mean(dim=(0, 2))

        return _metric_reduction(
            correct, reduction=reduction, weights=weights, ignore_index=ignore_index
        )
    else:
        return F.mse_loss(y_pred, y_true, reduction=reduction)


def soft_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    dim=None,
    square_denom=True,
) -> Tensor:
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=dim)

    if square_denom:
        cardinalities = y_pred.square().sum(dim=dim) + y_true.square().sum(dim=dim)
    else:
        cardinalities = y_pred.sum(dim=dim) + y_true.sum(dim=dim)

    dice_score = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    return dice_score


def soft_jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    dim=None,
    square_denom=True,
) -> Tensor:
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=dim)

    if square_denom:
        cardinalities = y_pred.square().sum(dim=dim) + y_true.square().sum(dim=dim)
    else:
        cardinalities = y_pred.sum(dim=dim) + y_true.sum(dim=dim)

    union = cardinalities - intersection

    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: str = "mean",
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

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=(0, 2))
    cardinalities = (y_pred == 1.0).sum(dim=(0, 2)) + (y_true == 1.0).sum(dim=(0, 2))

    score = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)

    return _metric_reduction(
        score, reduction=reduction, weights=weights, ignore_index=ignore_index
    )


def jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: str = "mean",
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

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=(0, 2))
    cardinalities = (y_pred == 1.0).sum(dim=(0, 2)) + (y_true == 1.0).sum(dim=(0, 2))
    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    return _metric_reduction(
        score, reduction=reduction, weights=weights, ignore_index=ignore_index
    )


@delegates(to=jaccard_score)
def IoU(y_pred: Tensor, y_true: Tensor, **kwargs) -> Tensor:
    return jaccard_score(y_pred, y_true, **kwargs)
