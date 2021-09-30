from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor


BINARY_MODE = "binary"
MULTI_MODE = "multiclass"
ONEHOT_MODE = "onehot"
AUTO_MODE = "auto"
MODES = (None, BINARY_MODE, MULTI_MODE, ONEHOT_MODE, AUTO_MODE)


def hard_max(x):
    """
    argmax + onehot
    """
    N = len(x.shape)
    order = (0, N - 1, *[i for i in range(1, N - 1)])
    return F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).permute(order)


def _inputs_as_onehot(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
    from_logits: bool = False,
    discretize: bool = False,
) -> Tuple[Tensor, Tensor]:
    assert mode in MODES
    batch_size, num_classes = y_pred.shape[:2]

    if mode == AUTO_MODE:
        if y_pred.shape == y_true.shape:
            mode = BINARY_MODE if num_classes == 1 else ONEHOT_MODE
        else:
            mode = MULTI_MODE

    if from_logits:
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1

        if mode == BINARY_MODE:
            y_pred = F.logsigmoid(y_pred.float()).exp()
        elif mode in (MULTI_MODE, ONEHOT_MODE):
            y_pred = F.log_softmax(y_pred.float(), dim=1).exp()

    if discretize:
        if mode == BINARY_MODE:
            y_pred = torch.round(y_pred).clamp_min(0.0).clamp_max(1.0)
        else:
            y_pred = hard_max(y_pred)

    if mode == BINARY_MODE:
        y_true = y_true.view(batch_size, 1, -1)
        y_pred = y_pred.view(batch_size, 1, -1)

    elif mode == ONEHOT_MODE:
        y_true = y_true.view(batch_size, num_classes, -1)
        y_pred = y_pred.view(batch_size, num_classes, -1)

    elif mode == MULTI_MODE:
        y_pred = y_pred.view(batch_size, num_classes, -1)
        y_true = y_true.view(batch_size, -1)
        y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

    assert y_pred.shape == y_true.shape
    return y_pred.float(), y_true.float()


def _inputs_as_longlabels(
    y_pred: Tensor,
    y_true: Tensor,
    mode: str = "auto",
    from_logits: bool = False,
    discretize: bool = False,
) -> Tuple[Tensor, Tensor]:

    batch_size, num_classes = y_pred.shape[:2]

    if mode == AUTO_MODE:
        if y_pred.shape == y_true.shape:
            mode = BINARY_MODE if num_classes == 1 else ONEHOT_MODE
        else:
            mode = MULTI_MODE

    if discretize:
        if mode == BINARY_MODE:
            y_pred = torch.round(y_pred).clamp_min(0.0).clamp_max(1.0)
        else:
            y_pred = hard_max(y_pred)

    if mode == BINARY_MODE:
        if from_logits:
            y_pred = F.logsigmoid(y_pred.float()).exp()
        y_pred = torch.round(y_pred).clamp_max(1).clamp_min(0).long()
    else:
        if from_logits:
            y_pred = F.log_softmax(y_pred.float(), dim=1).exp()
        batch_size, n_classes = y_pred.shape[:2]
        y_pred = y_pred.view(batch_size, n_classes, -1)
        y_pred = torch.argmax(y_pred, dim=1)

        if mode == ONEHOT_MODE:
            y_true = torch.argmax(y_true, dim=1)
        y_true = y_true.view(batch_size, -1)

    assert y_pred.shape == y_true.shape
    return y_pred, y_true.long()


def _metric_reduction(
    loss: Tensor,
    reduction: str = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:

    assert (
        weights is None or ignore_index is None
    ), "When setting weights, do not include ignore_index separately"

    if ignore_index is not None:
        weights = [1.0 if i != ignore_index else 0.0 for i in range(len(loss))]

    if weights:
        assert len(weights) == len(
            loss
        ), f"Weights must match number of channels {len(loss)} != {len(loss)}"

        if isinstance(weights, list):
            weights = torch.Tensor(weights)
        loss *= weights.type(loss.dtype).to(loss.device)

    N = len(loss)
    if ignore_index and 0 <= ignore_index < N:
        N -= 1
    if reduction is None:
        return loss
    if reduction == "mean":
        return loss.sum() / N
    if reduction == "sum":
        return loss.sum()
    if reduction == "batchwise_mean":
        return loss.sum(dim=0) / N


def batch_channel_flatten(x: Tensor) -> Tensor:
    batch_size, n_channels, *_ = x.shape
    return x.view(batch_size, n_channels, -1)
