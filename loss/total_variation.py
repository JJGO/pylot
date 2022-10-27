from typing import Optional

import torch
from torch import Tensor

from .util import _loss_module_from_func


def total_variation_loss(
    y_pred: Tensor,
    y_true: Optional[Tensor] = None,
    penalty: str = "l1",
    reduction: str = "mean",
):

    assert reduction in ("sum", "mean", None)
    assert penalty in ("l1", "l2",)
    shape = y_pred.shape[2:]
    ndims = len(shape)

    s_all = slice(None, None, None)  # [:]
    s_front = slice(None, -1, None)  # [:-1]
    s_back = slice(1, None, None)  # [1:]

    dxs = [
        y_pred[[s_back if i == j else s_all for j in range(-2, ndims)]]
        - y_pred[[s_front if i == j else s_all for j in range(-2, ndims)]]
        for i in range(ndims)
    ]

    if penalty == "l1":
        dxs = [torch.abs(dx) for dx in dxs]
    elif penalty == "l2":
        dxs = [torch.square(dx) for dx in dxs]

    if reduction is None:
        return dxs
    if reduction == "sum":
        return sum(dx.sum() for dx in dxs)
    if reduction == "mean":
        return sum(dx.mean() for dx in dxs) / ndims


TotalVariationLoss = _loss_module_from_func("TotalVariationLoss", total_variation_loss)
