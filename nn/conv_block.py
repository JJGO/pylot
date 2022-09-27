from dataclasses import dataclass
from typing import Optional, Any, Union, Literal, List

import torch.nn as nn

from .nonlinearity import get_nonlinearity
from .init import initialize_layer
from .norm import get_normlayer
from .drop import DropPath
from ..util.validation import validate_arguments_init


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvBlock(nn.Module):

    in_channels: int
    filters: List[int]
    kernel_size: Union[int, List[int]] = 3
    norm: Literal[None, "batch", "layer", "instance", "group"] = None
    activation: Optional[str] = "LeakyReLU"
    residual: bool = False
    drop_path: float = 0.0
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[float, int] = 0.0
    dims: Literal[1, 2, 3] = 2

    def __post_init__(self):
        super().__init__()

        conv_fn = getattr(nn, f"Conv{self.dims}d")
        nonlinearity_fn = get_nonlinearity(self.activation)

        self.F = nn.Sequential()

        all_channels = [self.in_channels] + self.filters
        for i, (n_in, n_out) in enumerate(zip(all_channels, all_channels[1:])):
            conv = conv_fn(
                n_in,
                n_out,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode="zeros",
            )
            self.F.add_module(f"n{i}_conv{self.dims}d", conv)

            if self.activation is not None:
                self.F.add_module(f"n{i}_act", nonlinearity_fn())

            if self.norm is not None:
                norm_layer = get_normlayer(n_out, self.norm, dims=self.dims)
                self.F.add_module(f"n{i}_{self.norm}norm", norm_layer)

        if self.residual:
            self.shortcut = nn.Sequential()
            if self.in_channels != self.filters[-1]:
                # When channels mismatch, residual op is y= F(x; W_i) + W_s x
                # Where W_s is a simple matmul that can be implemented with a matmul
                conv = conv_fn(
                    self.in_channels,
                    self.filters[-1],
                    kernel_size=1,
                )
                self.shortcut.add_module("conv", conv)

                if self.norm is not None:
                    norm_layer = get_normlayer(
                        self.filters[-1], self.norm, dims=self.dims
                    )
                    self.shortcut.add_module(f"{self.norm}norm", norm_layer)

            if self.drop_path > 0:
                self.shortcut.add_module("drop_path", DropPath(self.drop_path))

        self.reset_parameters()

    def forward(self, input):
        x = self.F(input)
        if self.residual:
            x = x + self.shortcut(input)
        return x

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                initialize_layer(
                    module,
                    distribution=self.init_distribution,
                    init_bias=self.init_bias,
                    nonlinearity=self.activation,
                )
