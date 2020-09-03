import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from .resnet_nopool import remove_first_maxpool

__all__ = [
    "SemiResNet",
    "semiresnet18",
    "semiresnet34",
    "semiresnet50",
    "semiresnet101",
    "semiresnet152",
    "semiresnext50_32x4d",
    "semiresnext101_32x8d",
    "wide_semiresnet50_2",
    "wide_semiresnet101_2",
]


class SemiResNet(ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        drop_first_maxpool=False,
    ):
        super(SemiResNet, self).__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        # Halve the size of the channels
        self.inplanes = 32
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(
            block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        if drop_first_maxpool:
            remove_first_maxpool(self)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = SemiResNet(block, layers, **kwargs)
    if pretrained:
        raise NotImplementedError("No pretrained models are available")
    return model


def semiresnet18(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "semiresnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def semiresnet34(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "semiresnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def semiresnet50(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "semiresnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def semiresnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "semiresnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def semiresnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "semiresnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def semiresnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "semiresnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def semiresnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "semiresnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs
    )


def wide_semiresnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_semiresnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_semiresnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_semiresnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs
    )
