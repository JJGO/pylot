from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from torch import nn, Tensor
import torch.nn.functional as F

from ..nn import ConvBlock
from ..nn import resize


@dataclass(eq=False, repr=False)
class ConvEncoder(nn.Module):

    in_channels: int
    filters: List[int]
    convs_per_block: int = 1
    dims: int = 2
    flatten: bool = False
    conv_kws: Optional[Dict[str, Any]] = None
    final_pool: bool = False

    def __post_init__(self):
        super().__init__()

        conv_kws = {"dims": self.dims}
        if self.conv_kws:
            conv_kws.update(self.conv_kws)

        self.downsample = getattr(nn, f"MaxPool{self.dims}d")(2)

        self.layers = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(
            zip([self.in_channels] + self.filters, self.filters)
        ):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_kws)
            self.layers.append(c)

    def forward(self, x: Tensor) -> Tensor:
        for i, block in enumerate(self.layers):
            x = block(x)
            if self.final_pool or i < len(self.layers) - 1:
                x = self.downsample(x)

        if self.flatten:
            x = x.view(x.size(0), -1)
        return x


@dataclass(eq=False, repr=False)
class ConvDecoder(nn.Module):

    in_channels: int
    out_channels: int
    filters: List[int]
    convs_per_block: int = 1
    dims: int = 2
    conv_kws: Optional[Dict[str, Any]] = None
    first_upsample: bool = False
    out_activation: Optional[str] = None

    def __post_init__(self):
        super().__init__()

        conv_kws = {"dims": self.dims}
        if self.conv_kws:
            conv_kws.update(self.conv_kws)

        self.layers = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(
            zip([self.in_channels] + self.filters, self.filters)
        ):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_kws)
            self.layers.append(c)

        self.out_conv = ConvBlock(
            self.filters[-1],
            [self.out_channels],
            activation=self.out_activation,
            kernel_size=1,
            dims=self.dims,
            batch_norm=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, block in enumerate(self.layers):
            if self.first_upsample or i > 0:
                x = resize(x, scale_factor=2)
            x = block(x)

        x = self.out_conv(x)

        return x
