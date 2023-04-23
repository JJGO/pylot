import numpy as np

from torch import nn

from ...nn import HookedModule
from .abstract_flops import convNd_flops, dense_flops


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


def _relu_flops(module, input):
    return input.numel()


def measure_flops(model, *inputs, **kw_inputs):

    _FLOP_fn = {
        nn.Conv2d: _convNd_flops,
        nn.Linear: _linear_flops,
        nn.LeakyReLU: _relu_flops,
    }
    rows = []

    def _store_flops(module, inputs, outputs):
        flops = _FLOP_fn[type(module)](module, inputs[0])
        rows.append(flops)

    with HookedModule(model, _store_flops, module_types=tuple(_FLOP_fn)):
        model(*inputs, **kw_inputs)

    return sum(rows)
