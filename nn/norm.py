from typing import Literal, Union, Tuple
from pydantic import validate_arguments

import torch
from torch import nn
import einops as E

from pylot.nn import batch_renorm

NormType = Union[
    Literal["batch", "batchre", "layer", "instance", "channel"],
    Tuple[Literal["group"], int],
]


class ChannelNorm(nn.LayerNorm):
    # Normalizes the C dimension in N C * tensors
    # This is the LayerNorm in ConvNext Networks

    def __init__(self, num_features, eps=1e-6, affine=True):
        super().__init__(num_features, eps=eps, elementwise_affine=affine)
        self.num_features = num_features

    @property
    def affine(self):
        return self.elementwise_affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = E.rearrange(x, "B C ... -> B ... C")
        super().forward(x)
        x = E.rearrange(x, "B ... C -> B C ...")
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_features={self.num_features}, eps={self.eps}, affine={self.affine})"


def get_normlayer(
    features: int,
    kind: NormType,
    dims: Literal[1, 2, 3] = 2,
    **norm_kws,
):
    if kind == "batch":
        return getattr(nn, f"BatchNorm{dims}d")(features, **norm_kws)
    if kind == "batchre":
        return getattr(batch_renorm, f"BatchRenorm{dims}d")(features, **norm_kws)
    if kind == "instance":
        return getattr(nn, f"InstanceNorm{dims}d")(features, **norm_kws)
    if kind == "layer":
        return nn.GroupNorm(1, features, **norm_kws)
    if kind == "channel":
        return ChannelNorm(features, **norm_kws)
    if isinstance(kind, tuple):
        _, groups = kind
        return nn.GroupNorm(groups, features, **norm_kws)
