from typing import Union, Tuple

import numpy as np
import kornia.augmentation as KA


# TODO: make typing work for float too
def _as2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    # because kornia.morphology works only with two-tuples
    if isinstance(value, (int, float)):
        return (value, value)
    if isinstance(value, list):
        value = tuple(value)
    assert isinstance(value, tuple) and len(value) == 2, f"Invalid 2-tuple {value}"
    return value


def _as_single_val(value):
    if isinstance(value, (int, float)):
        return value
    assert (
        isinstance(value, (tuple, list)) and len(value) == 2
    ), f"Invalid 2-tuple {value}"
    if any(isinstance(i, float) for i in value):
        value = (float(value[0]), float(value[1]))
    if isinstance(value[0], int):
        return np.random.randint(*value)
    else:
        return np.random.uniform(*value)


class AugmentationBase2D(KA.AugmentationBase2D):

    """ Dummy class because Kornia really wants me to overload
    the .compute_transformation method
    """

    def compute_transformation(self, input, params):
        return self.identity_matrix(input)
