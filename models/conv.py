import copy
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from torch import nn, Tensor
import torch.nn.functional as F

from ..util import validate_arguments_init
from ..nn import ConvBlock, resize, Flatten


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvEncoder(nn.Sequential):

    in_channels: int
    filters: List[int]
    convs_per_block: int = 1
    conv_kws: Optional[Dict[str, Any]] = None
    final_pool: bool = False
    dims: int = 2
    collapse_spatial: bool = False

    def __post_init__(self):
        super().__init__()

        conv_kws = {"dims": self.dims, **(self.conv_kws or {})}
        pool_fn = getattr(nn, f"MaxPool{self.dims}d")

        for i, (in_ch, out_ch) in enumerate(
            zip([self.in_channels] + self.filters, self.filters)
        ):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_kws)
            self.add_module(f"b{i}", c)
            if self.final_pool or i < len(self.filters) - 1:
                self.add_module(f"pool{i}", pool_fn(2))

        if self.collapse_spatial:
            self.gpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = Flatten()  # Flatten only to squeeze spatial dims


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvClassifier(nn.Sequential):

    in_channels: int
    n_classes: int
    filters: List[int]
    encoder_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()
        encoder_kws = self.encoder_kws or {}
        encoder_kws = {'collapse_spatial': True, **encoder_kws}
        self.encoder = ConvEncoder(self.in_channels, self.filters, **encoder_kws)
        self.flatten = Flatten()
        self.fc = nn.Linear(self.filters[-1], self.n_classes)

    def change_n_classes(self, n_classes):
        module = copy.deepcopy(self)
        module.fc = nn.Linear(module.filters[-1], n_classes)
        return module

    def encode(self, input):
        return self.flatten(self.gpool(self.encoder(input)))


@validate_arguments_init
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
