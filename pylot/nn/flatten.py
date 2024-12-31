from typing import Sequence, Iterable, Tuple

import torch
from torch import nn
from torch import Tensor


class Flatten(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.view(input.size(0), -1)


def flatten_tensors(tensors: Sequence[Tensor]) -> Tensor:
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat: Tensor, tensors: Iterable[Tensor]) -> Tuple[Tensor, ...]:
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def copy_unflattened(flat: Tensor, tensors: Iterable[Tensor]):
    for src, dst in zip(unflatten_tensors(flat, tensors), tensors):
        dst.data = src


class View(nn.Module):
    def __init__(self, shape, batch=True):
        super().__init__()
        self.shape = shape
        if batch:
            self.shape = (-1,) + self.shape

    def forward(self, x):
        return x.view(*self.shape)
