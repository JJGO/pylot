from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor

from ...nn.init import initialize_layer
from ...nn.nonlinearity import get_nonlinearity

from .encoder import Encoder

Shapes = Dict[str, Tuple[int, ...]]


@dataclass(eq=False, repr=False)
class HyperNet(nn.Module):

    input_sizes: Shapes
    output_sizes: Shapes
    layer_sizes: List[int]
    activation: str = "LeakyReLU"
    encoder: Optional[str] = None
    init_distribution: Optional[str] = None
    init_bias: Union[float, str] = 0.0
    rescale_output: bool = False  # TODO

    def __post_init__(self, reset=True):

        super().__init__()
        self.output_sizes = dict(self.output_sizes)

        nonlinearity = get_nonlinearity(self.activation)

        self._encoder = Encoder(mode=self.encoder)

        # Compute input size of MLP
        self.flat_input_size = sum(int(np.prod(v)) for v in self.input_sizes.values())
        self.flat_input_size *= self._encoder.factor

        offset = 0
        self.offsets = {}
        for name, shape in self.output_sizes.items():
            size = int(np.prod(shape))
            self.offsets[name] = (offset, size)
            offset += size
        self.flat_output_size = offset

        # Assign linear layers
        sizes = [self.flat_input_size] + self.layer_sizes
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            lin = nn.Linear(in_size, out_size)

            layers.extend(
                [
                    lin,
                    nonlinearity(),
                ]
            )

        layers.append(nn.Linear(self.layer_sizes[-1], self.flat_output_size))
        self.layers = nn.Sequential(*layers)

        if reset:
            self.reset_parameters()

    def reset_parameters(self):
        last = len(self.layers) - 1
        for i, module in enumerate(self.layers):
            if isinstance(module, nn.Linear):
                initialize_layer(
                    module,
                    self.init_distribution,
                    init_bias=self.init_bias,
                    nonlinearity=None if i == last else self.activation,
                )

    def _validate_input(self, inputs):
        assert set(inputs) == set(
            self.input_sizes
        ), f"Provided keys were {set(inputs)}, expected {set(self.input_sizes)}"
        for hp, shape in self.input_sizes.items():
            assert (
                inputs[hp].shape == shape
            ), f"Wrong shape for {hp}, expected {shape}, got {inputs[hp].shape}"

    def _flatten_input(self, inputs):

        # Linear layers expect Batch size by default. In our case this is 1
        flat_input = torch.cat(
            [inputs[hp].view(1, -1) for hp in sorted(self.input_sizes)], dim=-1
        )
        return flat_input

    def _unflatten_output(self, flat_output):
        outputs = {}
        for name, shape in self.output_sizes.items():
            outputs[name] = flat_output.narrow(1, *self.offsets[name]).view(shape)
        return outputs

    def forward(self, **inputs: Dict[str, Tensor]):
        self._validate_input(inputs)
        flat_input = self._flatten_input(inputs)
        flat_input = self._encoder(flat_input)
        flat_output = self.layers(flat_input)
        return self._unflatten_output(flat_output)
