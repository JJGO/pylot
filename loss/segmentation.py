from typing import Optional, Union, Literal

import torch
from torch import Tensor

from pydantic import validate_arguments

from .util import _loss_module_from_func
from ..util.more_functools import partial
from ..metrics.segmentation import soft_dice_score, soft_jaccard_score, pixel_mse
from ..metrics.util import InputMode, Reduction


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:

    score = soft_dice_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=weights,
        ignore_index=ignore_index,
        from_logits=from_logits,
        smooth=smooth,
        eps=eps,
        square_denom=square_denom,
    )

    if log_loss:
        loss = -torch.log(score.clamp_min(eps))
    else:
        loss = 1.0 - score

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_jaccard_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:

    score = soft_jaccard_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=weights,
        ignore_index=ignore_index,
        eps=eps,
        smooth=smooth,
        square_denom=square_denom,
    )

    if log_loss:
        loss = -torch.log(score.clamp_min(eps))
    else:
        loss = 1.0 - score

    return loss


pixel_mse_loss = pixel_mse
binary_soft_dice_loss = partial(soft_dice_loss, mode="binary")
binary_soft_jaccard_loss = partial(soft_jaccard_loss, mode="binary")

SoftDiceLoss = _loss_module_from_func("SoftDiceLoss", soft_dice_loss)
SoftJaccardLoss = _loss_module_from_func("SoftJaccardLoss", soft_jaccard_loss)
PixelMSELoss = _loss_module_from_func("PixelMSELoss", pixel_mse_loss)
