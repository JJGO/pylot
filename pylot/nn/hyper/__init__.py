# Basic Building blocks for VoidModules
from .xparam import ExternalParameter
from .void import VoidModule

# Void reimplementations of nn modules
from .linear import VoidLinear
from .conv import VoidConv1d, VoidConv2d, VoidConv3d
from .separable import (
    VoidDepthWiseSeparableConv1d,
    VoidDepthWiseSeparableConv2d,
    VoidDepthWiseSeparableConv3d,
)
from .conv_block import VoidConvBlock
