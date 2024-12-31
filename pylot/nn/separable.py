from typing import Union, Tuple
from torch import nn

from functools import partial, wraps

# Based on this: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch

__all__ = [
    "DepthWiseSeparableConv1d",
    "DepthWiseSeparableConv2d",
    "DepthWiseSeparableConv3d",
]


def _depthwise_helper(conv_fn):
    @wraps(conv_fn)
    def _depthwise_conv(in_channels, out_channels, **kwargs):
        return conv_fn(in_channels, out_channels, groups=in_channels, **kwargs)

    return _depthwise_conv


DepthwiseConv1d = _depthwise_helper(nn.Conv1d)
DepthwiseConv2d = _depthwise_helper(nn.Conv2d)
DepthwiseConv3d = _depthwise_helper(nn.Conv3d)

PointwiseConv1d = partial(nn.Conv1d, kernel_size=1)
PointwiseConv2d = partial(nn.Conv2d, kernel_size=1)
PointwiseConv3d = partial(nn.Conv3d, kernel_size=1)


class DepthWiseSeparableConvND(nn.Module):
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
        shared_conv_kwargs = dict(bias=bias, padding_mode=padding_mode,)

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


class DepthWiseSeparableConv1d(DepthWiseSeparableConvND):
    _conv_fn = nn.Conv1d


class DepthWiseSeparableConv2d(DepthWiseSeparableConvND):
    _conv_fn = nn.Conv2d


class DepthWiseSeparableConv3d(DepthWiseSeparableConvND):
    _conv_fn = nn.Conv3d
