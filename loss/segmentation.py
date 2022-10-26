from functools import partial
from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from .util import _loss_module_from_func
from ..util.more_functools import auto_assign_arguments
from ..metrics import soft_dice_score, soft_jaccard_score, pixel_mse
from ..metrics.util import _metric_reduction, _inputs_as_onehot, BINARY_MODE


def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
    reduction: str = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:
    """
    Note: _inputs_as_onehot() will convert inputs from (N, C, H, W) to (N, C, H*W)
    """
    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    dim = (0, *range(2, len(y_pred.shape)))
    scores = soft_dice_score(
        y_pred, y_true, smooth=smooth, eps=eps, dim=dim, square_denom=square_denom
    )

    if log_loss:
        loss = -torch.log(scores.clamp_min(eps))
    else:
        loss = 1.0 - scores

    return _metric_reduction(
        loss, reduction=reduction, weights=weights, ignore_index=ignore_index
    )


def soft_jaccard_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
    reduction: str = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    dim = (0, *range(2, len(y_pred.shape)))
    scores = soft_jaccard_score(
        y_pred, y_true, smooth=smooth, eps=eps, dim=dim, square_denom=square_denom
    )

    if log_loss:
        loss = -torch.log(scores.clamp_min(eps))
    else:
        loss = 1.0 - scores

    return _metric_reduction(
        loss, reduction=reduction, weights=weights, ignore_index=ignore_index
    )


pixel_mse_loss = pixel_mse
binary_soft_dice_loss = partial(soft_dice_loss, mode=BINARY_MODE)
binary_soft_jaccard_loss = partial(soft_jaccard_loss, mode=BINARY_MODE)

SoftDiceLoss = _loss_module_from_func("SoftDiceLoss", soft_dice_loss)
SoftJaccardLoss = _loss_module_from_func("SoftJaccardLoss", soft_jaccard_loss)
PixelMSELoss = _loss_module_from_func("PixelMSELoss", pixel_mse_loss)

# class SoftDiceLoss(nn.Module):
#     @auto_assign_arguments
#     def __init__(
#         self,
#         mode: str = "auto",
#         reduction: str = "mean",
#         weights: Optional[Union[Tensor, List]] = None,
#         ignore_index: Optional[int] = None,
#         from_logits: bool = False,
#         smooth: float = 1e-7,
#         eps: float = 1e-7,
#         square_denom: bool = True,
#         log_loss: bool = False,
#     ):
#         super().__init__()

#     def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
#         return soft_dice_loss(
#             y_pred,
#             y_true,
#             mode=self.mode,
#             weights=self.weights,
#             ignore_index=self.ignore_index,
#             from_logits=self.from_logits,
#             smooth=self.smooth,
#             eps=self.eps,
#             square_denom=self.square_denom,
#             log_loss=self.log_loss,
#         )


# class SoftJaccardLoss(nn.Module):
#     @auto_assign_arguments
#     def __init__(
#         self,
#         mode: str = "auto",
#         reduction: str = "mean",
#         weights: Optional[Union[Tensor, List]] = None,
#         ignore_index: Optional[int] = None,
#         from_logits: bool = False,
#         smooth: float = 1e-7,
#         eps: float = 1e-7,
#         square_denom: bool = True,
#         log_loss: bool = False,
#     ):
#         super().__init__()

#     def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
#         return soft_jaccard_loss(
#             y_pred,
#             y_true,
#             mode=self.mode,
#             weights=self.weights,
#             ignore_index=self.ignore_index,
#             from_logits=self.from_logits,
#             smooth=self.smooth,
#             eps=self.eps,
#             square_denom=self.square_denom,
#             log_loss=self.log_loss,
#         )
