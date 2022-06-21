from torch import nn

from . import separable
from ...nn.nonlinearity import get_nonlinearity
from ...nn.init import initialize_layer

from typing import List, Optional, Union


class ConvBlock(nn.Module):
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
        init_distribution: Optional[str] = "kaiming_normal",
        init_bias: Union[float, str] = 0.0,
    ):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.init_distribution = init_distribution
        self.init_bias = init_bias

        if activation is not None:
            nonlinearity = get_nonlinearity(activation)

        conv_fn = getattr(nn, f"Conv{dims}d")
        if depthsep:
            conv_fn = getattr(separable, f"DepthWiseSeparableConv{dims}d")
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
            self.shortcut = getattr(nn, f"Conv{dims}d")(
                inplanes, filters[-1], kernel_size=1
            )

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                initialize_layer(
                    module,
                    distribution=self.init_distribution,
                    init_bias=self.init_bias,
                    nonlinearity=self.activation,
                )

    def forward(self, input):
        x = self.F(input)
        if self.residual:
            if self.shortcut:
                input = self.shortcut(input)
            x = x + input
        return x
