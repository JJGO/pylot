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

        if activation is not None:
            nonlinearity = get_nonlinearity(activation)

        conv_fn = getattr(nn, f"Conv{dims}d")
        if depthsep:
            conv_fn = getattr(separable, f"DepthWiseSeparableConv{dims}d")
        bn_fn = getattr(nn, f"BatchNorm{dims}d")

        ops = []
        for n_in, n_out in zip([inplanes] + filters, filters):
            conv = conv_fn(
                n_in,
                n_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode="zeros",
            )
            ops.append(conv)

            if activation is not None:
                ops.append(nonlinearity())
            if batch_norm:
                ops.append(bn_fn(n_out))
        self.f = nn.Sequential(*ops)

        self.aux = None
        if residual and inplanes != filters[-1]:
            self.aux = getattr(nn, f"Conv{dims}d")(inplanes, filters[-1], kernel_size=1)

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
        x = self.f(input)
        if self.residual:
            if self.aux:
                input = self.aux(input)
            x = x + input
        return x


# def ConvBlock(
#     inplanes,
#     filters,
#     dims=2,
#     activation="LeakyReLU",
#     batch_norm=True,
#     kernel_size=3,
#     depthsep=False,
#     residual=False,  # TODO add residual connections
# ):

#     if activation is not None:
#         nonlinearity = getattr(nn, activation)

#     conv_fn = getattr(nn, f"Conv{dims}d")
#     if depthsep:
#         conv_fn = getattr(separable, f"separable.DepthWiseSeparableConv{dims}d")
#     bn_fn = getattr(nn, f"BatchNorm{dims}d")

#     ops = []
#     for n_in, n_out in zip([inplanes] + filters, filters):
#         conv = conv_fn(
#             n_in,
#             n_out,
#             kernel_size=kernel_size,
#             padding=kernel_size // 2,
#             padding_mode="zeros",
#         )
#         ops.append(conv)

#         if activation is not None:
#             ops.append(nonlinearity())
#         if batch_norm:
#             ops.append(bn_fn(n_out))
#     return nn.Sequential(*ops)
