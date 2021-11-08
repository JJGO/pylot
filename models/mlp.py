from dataclasses import dataclass
from typing import List, Optional, Union

from torch import nn

from ..nn.init import initialize_layer


@dataclass(eq=False, repr=False)
class MLP(nn.Module):

    in_features: int
    out_features: int
    hidden: List[int]
    out_activation: Optional[int] = None
    activation: str = "LeakyReLU"
    dropout: Optional[float] = None
    batch_norm: bool = False
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[float, str] = 0.0

    def __post_init__(self):
        super().__init__()
        self.layers = nn.Sequential()

        nonlinearity = getattr(nn, self.activation)

        for i, (n_in, n_out) in enumerate(
            zip([self.in_features] + self.hidden, self.hidden + [self.out_features])
        ):
            self.layers.add_module(f"fc{i}", nn.Linear(n_in, n_out))
            if i < len(self.hidden):
                self.layers.add_module(f"act{i}", nonlinearity())
                if self.batch_norm:
                    self.layers.add_module(f"bn{i}", nn.BatchNorm1d(n_out))
                if self.dropout is not None:
                    self.layers.add_module(f"drop{i}", nn.Dropout(self.dropout))

        if self.out_activation:
            if self.out_activation == "Softmax":
                self.layers.add_module("out_act", nn.Softmax(self.out_features))
            else:
                self.layers.add_module("out_act", getattr(nn, self.out_activation)())

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initialize_layer(
                    module,
                    distribution=self.init_distribution,
                    init_bias=self.init_bias,
                    nonlinearity=self.activation,
                )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x)

    def set_dropout(self, rate: float):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
