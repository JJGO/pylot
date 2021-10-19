from typing import Union, Tuple

from .void import VoidModule
from .conv import VoidConv1d, VoidConv2d, VoidConv3d

from functools import partial, wraps

# Based on this: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch


def _depthwise_helper(conv_fn):
    @wraps(conv_fn)
    def _depthwise_conv(in_channels, out_channels, **kwargs):
        return conv_fn(in_channels, out_channels, groups=in_channels, **kwargs)

    return _depthwise_conv


VoidDepthwiseConv1d = _depthwise_helper(VoidConv1d)
VoidDepthwiseConv2d = _depthwise_helper(VoidConv2d)
VoidDepthwiseConv3d = _depthwise_helper(VoidConv3d)

VoidPointwiseConv1d = partial(VoidConv1d, kernel_size=1)
VoidPointwiseConv2d = partial(VoidConv2d, kernel_size=1)
VoidPointwiseConv3d = partial(VoidConv3d, kernel_size=1)


class DepthWiseSeparableConvND(VoidModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        kernels_per_layer: int = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[str, int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels_per_layer = kernels_per_layer
        self.mid_channels = in_channels * kernels_per_layer
        shared_conv_kwargs = dict(
            bias=bias,
            padding_mode=padding_mode,
        )

        self.depthwise = self._conv_fn(
            in_channels,
            self.mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            **shared_conv_kwargs
        )
        self.pointwise = self._conv_fn(
            self.mid_channels, out_channels, kernel_size=1, **shared_conv_kwargs
        )

    def forward(self, input):
        x = self.depthwise(input)
        x = self.pointwise(x)
        return x

    @property
    def kernel_size(self):
        return self.depthwise.kernel_size

    @property
    def stride(self):
        return self.depthwise.stride

    @property
    def padding(self):
        return self.depthwise.padding

    @property
    def dilation(self):
        return self.depthwise.dilation

    @property
    def padding_mode(self):
        return self.depthwise.padding_mode


class VoidDepthWiseSeparableConv1d(DepthWiseSeparableConvND):
    _conv_fn = VoidConv1d


class VoidDepthWiseSeparableConv2d(DepthWiseSeparableConvND):
    _conv_fn = VoidConv2d


class VoidDepthWiseSeparableConv3d(DepthWiseSeparableConvND):
    _conv_fn = VoidConv3d
