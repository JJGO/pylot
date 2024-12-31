from typing import Callable, Tuple, Union

import numpy as np

import torch
from torch import nn

Axes = Union[int, Tuple[int, ...]]

# Jaxlike vmap exploiting reshaping to the batch dimension.
# Applies the given module


def bmap(module: Callable, *xs: torch.Tensor, axes: Axes = 1, **kwargs):
    # Pre-process axes
    if isinstance(axes, int):
        axes = (axes,)

    if 0 in axes:
        raise ValueError("Batch dimension (0) cannot be included in axes")

    # Normalize negative axes
    N = len(xs[0].shape)
    axes = tuple(i if i > 0 else N + i for i in axes)

    # Ensure all inputs same number of dims
    if len(set([x.shape for x in xs])) > 1:
        msg = "; ".join(f"{tuple(x.shape)}" for x in xs)
        raise ValueError(f"Inputs have mismatched number of dimensions: {msg}")

    # Compute permutation & shapes
    batch_axes = (0,) + tuple(sorted(axes))
    other_axes = tuple(i for i in range(len(xs[0].shape)) if i not in batch_axes)
    permutation = batch_axes + other_axes
    inverse_permutation = tuple(np.argsort(permutation).tolist())

    original_shape = xs[0].shape
    batch_dim = np.prod([original_shape[dim] for dim in batch_axes])
    batched_shape = (batch_dim,) + tuple([original_shape[dim] for dim in other_axes])

    # Permute, reshape & apply fn
    batched_inputs = [x.permute(permutation).reshape(batched_shape) for x in xs]
    batched_output = module(*batched_inputs, **kwargs)

    def undo(x: torch.Tensor):
        unbatched_shape = tuple([original_shape[dim] for dim in batch_axes]) + tuple(
            x.shape[1:]
        )
        return x.reshape(unbatched_shape).permute(inverse_permutation)

    if isinstance(batched_output, tuple):
        return tuple(map(undo, batched_output))

    return undo(batched_output)


def bmap_fn(fn: Callable, axes: Axes = 1):
    def bmapped_fn(*args, **kwargs):
        return bmap(fn, *args, axes=axes, **kwargs)

    return bmapped_fn


class Bmap(nn.Module):
    def __init__(self, module: nn.Module, axes: Axes = 1):
        super().__init__()
        self.axes = axes
        self.module = module

    def forward(self, x: torch.Tensor):
        return bmap(self.module, x, axes=self.axes)


def bmap_cls(module_type: type, axes: Axes = 1):
    def bmapped_cls(*args, **kwargs):
        module = module_type(*args, **kwargs)
        return Bmap(module, axes)

    return bmapped_cls
