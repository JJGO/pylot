from typing import Literal
from pydantic import validate_arguments


@validate_arguments
def get_normlayer(
    features: int,
    kind: Literal["batch", "layer", "instance", "group"],
    dims: Literal[1, 2, 3] = 2,
):
    raise NotImplementedError()
