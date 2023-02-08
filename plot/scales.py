import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np


class Log2Scale(mscale.LogScale):

    name = "log2"
    """
    Scale based on logarithm to base 2.
    """

    def __init__(self, axis, **kwargs):
        super().__init__(axis, base=2, **kwargs)


mscale.register_scale(Log2Scale)


class SymmetricalLog2Scale(mscale.SymmetricalLogScale):

    name = "symlog2"
    """
    Scale based on logarithm to base 2.
    """

    def __init__(self, axis, **kwargs):
        super().__init__(axis, base=2, **kwargs)


mscale.register_scale(SymmetricalLog2Scale)
