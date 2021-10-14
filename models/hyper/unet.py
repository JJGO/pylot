import copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from pylot.nn import resize


@dataclass(eq=False, repr=False)
class VoidUNet(nn.Module):
    in_channels: int
    out_channels: int
    filters: List[int]
    up_filters: Optional[List[int]] = None
    activation: str = "LeakyReLU"
    out_activation: Optional[str] = None
    convs_per_block: int = 1
    batch_norm: bool = True
    parametrized: bool = False
    prune_init: float = 0.5
    skip_connections: bool = True
    interpolation_mode: str = "linear"
    dims: int = 2

    def __post_init__(self):
        super().__init__()

        filters = self.filters
        if self.up_filters is None:
            self.up_filters = filters[-2::-1]
        assert len(self.up_filters) == len(self.filters) - 1

        self.conv_args = dict(kernel_size=3, padding=3 // 2)

        self.shapes = {}
        self.layers = nn.ModuleDict()
        nonlinearity = getattr(nn, self.activation)

        conv_fn = getattr(nn, f"Conv{self.dims}d")
        bn_fn = getattr(nn, f"BatchNorm{self.dims}d")

        down_inputs = zip([self.in_channels] + filters[:-1], filters)
        for i, (in_ch, out_ch) in enumerate(down_inputs):
            fs = [in_ch] + [out_ch] * self.convs_per_block

            for j, (n_in, n_out) in enumerate(zip(fs, fs[1:])):
                conv = conv_fn(n_in, n_out, **self.conv_args)
                self.shapes[f"down_{i}_{j}_weight"] = conv.weight.shape
                self.shapes[f"down_{i}_{j}_bias"] = conv.bias.shape
                self.layers[f"down_{i}_{j}_nla"] = nonlinearity()
                if self.batch_norm:
                    self.layers[f"down_{i}_{j}_bn"] = bn_fn(n_out)

        prev_out_ch = filters[-1]
        skip_chs = filters[-2::-1]
        # for i, (in_ch, out_ch) in enumerate(zip(filters[::-1], self.up_filters)):
        for i, (skip_ch, out_ch) in enumerate(zip(skip_chs, self.up_filters)):
            in_ch = skip_ch + prev_out_ch if self.skip_connections else prev_out_ch
            fs = [in_ch] + [out_ch] * self.convs_per_block
            prev_out_ch = out_ch

            for j, (n_in, n_out) in enumerate(zip(fs, fs[1:])):
                conv = conv_fn(n_in, n_out, **self.conv_args)
                self.shapes[f"up_{i}_{j}_weight"] = conv.weight.shape
                self.shapes[f"up_{i}_{j}_bias"] = conv.bias.shape
                self.layers[f"up_{i}_{j}_nla"] = nonlinearity()
                if self.batch_norm:
                    self.layers[f"up_{i}_{j}_bn"] = bn_fn(n_out)

        # filters[0] == up_filters[-1]
        conv = conv_fn(prev_out_ch, self.out_channels, kernel_size=1)
        self.shapes["out_conv_weight"] = conv.weight.shape
        self.shapes["out_conv_bias"] = conv.bias.shape

        del self.conv_args["kernel_size"]

        if self.out_activation is not None:
            if self.out_activation == "Softmax":
                # For softmax we need to specify the channel dimension manually
                self.out_activation = nn.Softmax(dim=1)
            else:
                self.out_activation = getattr(nn, self.out_activation)()

    @property
    def parameter_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return copy.deepcopy(self.shapes)

    def forward(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:

        conv_outputs = []
        conv_fn = getattr(F, f"conv{self.dims}d")
        pool_fn = getattr(F, f"max_pool{self.dims}d")

        # Down Blocks
        for i, _ in enumerate(self.filters):
            for j in range(self.convs_per_block):
                w = params[f"down_{i}_{j}_weight"]
                b = params[f"down_{i}_{j}_bias"]
                x = conv_fn(x, w, b, **self.conv_args)
                x = self.layers[f"down_{i}_{j}_nla"](x)
                if self.batch_norm:
                    x = self.layers[f"down_{i}_{j}_bn"](x)
            if i == len(self.filters) - 1:
                break
            conv_outputs.append(x)
            x = pool_fn(x, 2)

        # Up blocks
        for i, _ in enumerate(self.up_filters):
            size = conv_outputs[-(i + 1)].size()[-self.dims :]
            x = resize(x, size=size, interpolation_mode=self.interpolation_mode)
            if self.skip_connections:
                x = torch.cat([x, conv_outputs[-(i + 1)]], dim=1)
            for j in range(self.convs_per_block):
                w = params[f"up_{i}_{j}_weight"]
                b = params[f"up_{i}_{j}_bias"]
                x = conv_fn(x, w, b, **self.conv_args)
                x = self.layers[f"up_{i}_{j}_nla"](x)
                if self.batch_norm:
                    x = self.layers[f"up_{i}_{j}_bn"](x)

        x = conv_fn(x, params["out_conv_weight"], params["out_conv_bias"])

        if self.out_activation is not None:
            x = self.out_activation(x)

        return x
