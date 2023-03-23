from dataclasses import dataclass
from typing import Optional, Union

from torch import nn

from ..util.validation import validate_arguments_init
from .init import reset_conv2d_parameters
from .nonlinearity import get_nonlinearity
from .norm import NormType, get_normlayer
from .types_ import size2t


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    norm: Optional[NormType] = None
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=self.norm is None,
            # normalizing will remove need for bias, it'll get shifted into the affine params
        )

        if self.norm is not None:
            self.norml = get_normlayer(self.out_channels, kind=self.norm, dims=2)

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)()

        reset_conv2d_parameters(self)
