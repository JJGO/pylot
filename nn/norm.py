from typing import Literal
from pydantic import validate_arguments

from torch import nn


@validate_arguments
def get_normlayer(
    features: int,
    kind: Literal["batch", "layer", "instance", "group"],
    dims: Literal[1, 2, 3] = 2,
):
    if kind == "batch" and dims == 2:
        return nn.BatchNorm2d(features)
    raise NotImplementedError()
