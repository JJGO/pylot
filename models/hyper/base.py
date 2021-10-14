import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings

import numpy as np
import torch
from torch import nn
from torch import Tensor

from ...layers.regularization import L2ActivationRegularizer
from ...nn.init import initialize_layer
from ...nn.nonlinearity import get_nonlinearity


@dataclass(eq=False, repr=False)
class HyperNet(nn.Module):

    input_sizes: Dict[str, Tuple[int, ...]]
    layer_sizes: List[int]
    output_sizes: Dict[str, Tuple[int, ...]]
    activation: str = "LeakyReLU"
    encoder: Optional[str] = None
    output_weight_decay: Optional[float] = None
    init_distribution: Optional[str] = None
    init_bias: Union[float, str] = 0.0

    def __post_init__(self):
        if self.init_kws is not None:
            warnings.warn(
                "init_kws is deprecated, please use init_distribution and init_zero_bias instead"
            )

        super().__init__()
        nonlinearity = get_nonlinearity(self.activation)

        self.flat_input_size = sum(int(np.prod(v)) for v in self.input_sizes.values())

        self.flat_input_size *= {
            None: 1,
            "1-z": 1,
            "(z, 1-z)": 2,
            "(z, -z)": 2,
            "(cos, sin)": 2,
            "(cos, sin)*": 2,
            "10+z": 1,
            "2z-1": 1,
            "1+z": 1,
            "z|z": 2,
        }[self.encoder]

        sizes = [self.flat_input_size] + self.layer_sizes
        layers = nn.ModuleList()
        self.heads = nn.ModuleDict()

        for in_size, out_size in zip(sizes, sizes[1:]):
            lin = nn.Linear(in_size, out_size)

            layers.extend(
                [
                    lin,
                    nonlinearity(),
                ]
            )

        for k, size in self.output_sizes.items():

            flat_size = np.prod(size)
            lin = nn.Linear(self.layer_sizes[-1], flat_size)

            if self.output_weight_decay is None:
                self.heads[k] = lin
            else:
                self.heads[k] = nn.Sequential(
                    lin, L2ActivationRegularizer(self.output_weight_decay)
                )

        self.backbone = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):

        for module in self.backbone:
            initialize_layer(
                module,
                self.init_distribution,
                init_bias=self.init_bias,
                nonlinearity=self.activation,
            )
        for module in self.heads.values():
            # Output is linear
            initialize_layer(
                module,
                self.init_distribution,
                init_bias=self.init_bias,
                nonlinearity=None,
            )

    def _validate_input(self, inputs):
        assert set(inputs) == set(
            self.input_sizes
        ), f"Provided keys were {set(inputs)}, expected {set(self.input_sizes)}"
        for hp, shape in self.input_sizes.items():
            assert (
                inputs[hp].shape == shape
            ), f"Wrong shape for {hp}, expected {shape}, got {inputs[hp].shape}"

    def _encode_input(self, flat_input):
        z = flat_input
        if self.encoder == "1-z":
            flat_input = 1 - z
        if self.encoder == "(z, 1-z)":
            flat_input = torch.cat([z, 1 - z], dim=-1)
        if self.encoder == "(z, -z)":
            flat_input = torch.cat([z, -z], dim=-1)
        if self.encoder == "(cos, sin)":
            a = 2 * z / math.pi
            flat_input = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        if self.encoder == "(cos, sin)*":
            a = math.pi * z / 2
            flat_input = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        if self.encoder == "10+z":
            flat_input = 10 + flat_input
        if self.encoder == "2z-1":
            flat_input = 2 * flat_input - 1
        if self.encoder == "1+z":
            flat_input = 1 + flat_input
        if self.encoder == "z|z":
            flat_input = torch.cat([z, z], dim=-1)

        return flat_input

    def _flatten_input(self, inputs):

        # Linear layers expect Batch size by default. In our case this is 1
        flat_input = torch.cat(
            [inputs[hp].view(1, -1) for hp in sorted(self.input_sizes)], dim=-1
        )
        return flat_input

    def forward(self, **inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:

        self._validate_input(inputs)
        flat_input = self._flatten_input(inputs)
        flat_input = self._encode_input(flat_input)
        intermediate = self.backbone(flat_input)

        outputs = {
            k: self.heads[k](intermediate).view(self.output_sizes[k])
            for k in self.heads
        }

        return outputs
