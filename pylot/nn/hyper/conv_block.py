from torch import nn

from . import separable
from ...nn.nonlinearity import get_nonlinearity
from ...nn.init import initialize_layer

from typing import List, Optional, Union

from .. import hyper

from .void import VoidModule


class VoidConvBlock(VoidModule):
    def __init__(
        self,
        inplanes: int,
        filters: List[int],
        dims: int = 2,
        kernel_size: int = 3,
        activation: str = "LeakyReLU",
        batch_norm: bool = True,
        residual: bool = False,
        depthsep: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.activation = activation

        if activation is not None:
            nonlinearity = get_nonlinearity(activation)

        conv_fn = getattr(hyper, f"VoidConv{dims}d")
        if depthsep:
            conv_fn = getattr(separable, f"VoidDepthWiseSeparableConv{dims}d")
        bn_fn = getattr(nn, f"BatchNorm{dims}d")

        self.F = nn.Sequential()
        for i, (n_in, n_out) in enumerate(zip([inplanes] + filters, filters)):
            conv = conv_fn(
                n_in,
                n_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode="zeros",
            )
            self.F.add_module(f"b{i}_conv", conv)

            if activation is not None:
                self.F.add_module(f"b{i}_act", nonlinearity())
            if batch_norm:
                self.F.add_module(f"b{i}_bn", bn_fn(n_out))
        self.shortcut = None
        if residual and inplanes != filters[-1]:
            self.shortcut = getattr(hyper, f"VoidConv{dims}d")(
                inplanes, filters[-1], kernel_size=1
            )

    def forward(self, input):
        x = self.F(input)
        if self.residual:
            if self.shortcut:
                input = self.shortcut(input)
            x = x + input
        return x
