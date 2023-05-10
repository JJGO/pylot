from enum import Enum
from typing import Literal, Optional, Tuple, Union

import einops as E
from pydantic import validate_arguments

import torch
import torch.nn.functional as F
from torch import Tensor

InputMode = Literal["binary", "multiclass", "onehot", "auto"]
Reduction = Union[None, Literal["mean", "sum"]]


def hard_max(x: Tensor):
    """
    argmax + onehot
    """
    N = len(x.shape)
    order = (0, N - 1, *[i for i in range(1, N - 1)])
    return F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).permute(order)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _infer_mode(y_pred: Tensor, y_true: Tensor,) -> InputMode:
    batch_size, num_classes = y_pred.shape[:2]

    if y_pred.shape == y_true.shape:
        return "binary" if num_classes == 1 else "onehot"
    else:
        return "multiclass"


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _inputs_as_onehot(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    discretize: bool = False,
) -> Tuple[Tensor, Tensor]:
    batch_size, num_classes = y_pred.shape[:2]

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    if from_logits:
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1

        if mode == "binary":
            # y_pred = F.logsigmoid(y_pred.float()).exp()
            y_pred = torch.sigmoid(y_pred.float())
        elif mode in ("multiclass", "onehot"):
            # y_pred = F.log_softmax(y_pred.float(), dim=1).exp()
            y_pred = torch.softmax(y_pred.float(), dim=1)

    if discretize:
        if mode == "binary":
            y_pred = torch.round(y_pred).clamp_min(0.0).clamp_max(1.0)
            y_true = torch.round(y_true).clamp_min(0.0).clamp_max(1.0)
        elif mode == "onehot":
            y_pred = hard_max(y_pred)
            y_true = hard_max(y_true)
        elif mode == "multiclass":
            y_pred = hard_max(y_pred)

    if mode == "binary":
        y_true = y_true.reshape(batch_size, 1, -1)
        y_pred = y_pred.reshape(batch_size, 1, -1)

    elif mode == "onehot":
        y_true = y_true.reshape(batch_size, num_classes, -1)
        y_pred = y_pred.reshape(batch_size, num_classes, -1)

    elif mode == "multiclass":
        y_pred = y_pred.reshape(batch_size, num_classes, -1)
        y_true = y_true.reshape(batch_size, -1)
        y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

    assert y_pred.shape == y_true.shape
    return y_pred.float(), y_true.float()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _inputs_as_longlabels(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    discretize: bool = False,
) -> Tuple[Tensor, Tensor]:

    batch_size, num_classes = y_pred.shape[:2]

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    if discretize:
        if mode == "binary":
            y_pred = torch.round(y_pred).clamp_min(0.0).clamp_max(1.0)
        else:
            y_pred = hard_max(y_pred)

    if mode == "binary":
        if from_logits:
            y_pred = F.logsigmoid(y_pred.float()).exp()
        y_pred = torch.round(y_pred).clamp_max(1).clamp_min(0).long()
    else:
        if from_logits:
            y_pred = F.log_softmax(y_pred.float(), dim=1).exp()
        batch_size, n_classes = y_pred.shape[:2]
        y_pred = y_pred.view(batch_size, n_classes, -1)
        y_pred = torch.argmax(y_pred, dim=1)

        if mode == "onehot":
            y_true = torch.argmax(y_true, dim=1)
        y_true = y_true.view(batch_size, -1)

    assert y_pred.shape == y_true.shape
    return y_pred, y_true.long()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _metric_reduction(
    loss: Tensor,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:

    if len(loss.shape) != 2:
        raise ValueError(
            f"Reduceable Tensor must have two dimensions, batch & channels, got {loss.shape} instead"
        )

    batch, channels = loss.shape

    assert (
        weights is None or ignore_index is None
    ), "When setting weights, do not include ignore_index separately"

    if ignore_index is not None:
        weights = [1.0 if i != ignore_index else 0.0 for i in range(channels)]

    if weights:
        assert (
            len(weights) == channels
        ), f"Weights must match number of channels {len(weights)} != {channels}"

        if isinstance(weights, list):
            weights = torch.Tensor(weights)
        weights = E.repeat(weights, "C -> B C", C=channels, B=batch)
        loss *= weights.type(loss.dtype).to(loss.device)

    if batch_reduction == "sum":
        loss = loss.sum(dim=0)
    elif batch_reduction == "mean":
        loss = loss.mean(dim=0)

    N = channels
    if ignore_index is not None and 0 <= ignore_index < N:
        N -= 1
    if reduction is None:
        return loss
    if reduction == "mean":
        return loss.sum(dim=-1) / N
    if reduction == "sum":
        return loss.sum(dim=-1)


def batch_channel_flatten(x: Tensor) -> Tensor:
    batch_size, n_channels, *_ = x.shape
    return x.view(batch_size, n_channels, -1)
