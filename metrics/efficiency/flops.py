import numpy as np
from torch import nn

from .abstract_flops import dense_flops, convNd_flops


def _convNd_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    # TODO Add support for dilation and padding size
    return convNd_flops(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        input_shape=input_shape,
        kernel_shape=module.kernel_size,
        padding=module.padding_mode,
        strides=module.stride,
        dilation=module.dilation,
    )


def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    return dense_flops(module.in_features, module.out_features)


def flops(model, input):
    """Compute Multiply-add FLOPs estimate from model

    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        input {torch.Tensor} -- Input tensor needed for activations

    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """
    FLOP_fn = {
        nn.Conv1d: _convNd_flops,
        nn.Conv2d: _convNd_flops,
        nn.Conv3d: _convNd_flops,
        nn.Linear: _linear_flops,
    }

    total_flops = 0
    activations = get_activations(model, input)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            total_flops += FLOP_fn[m.__class__](m, act)

    return total_flops
