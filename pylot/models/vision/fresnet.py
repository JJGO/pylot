import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

__all__ = [
    "FResNet",
    "fresnet18",
    "fresnet34",
    "fresnet50",
    "fresnet101",
    "fresnet152",
    "fresnext50_32x4d",
    "fresnext101_32x8d",
    "wide_fresnet50_2",
    "wide_fresnet101_2",
]


def remove_first_maxpool(resnet):
    assert isinstance(resnet, ResNet)
    resnet.maxpool = nn.Identity()
    return resnet


class FResNet(ResNet):
    def __init__(
        self,
        block,
        layers,
        filters=(64, 128, 256, 512),
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        drop_first_maxpool=False,
    ):
        super(FResNet, self).__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )
        self.filters = filters

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
        self.inplanes = filters[0]
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.layer1 = self._make_layer(block, filters[0], layers[0])
        self.layer2 = self._make_layer(
            block,
            filters[1],
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            filters[2],
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            filters[3],
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.fc = nn.Linear(filters[3] * block.expansion, num_classes)

        if drop_first_maxpool:
            remove_first_maxpool(self)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = FResNet(block, layers, **kwargs)
    if pretrained:
        raise NotImplementedError("No pretrained models are available")
    return model


def fresnet18(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "fresnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def fresnet34(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "fresnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def fresnet50(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "fresnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def fresnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "fresnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def fresnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "fresnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def fresnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "fresnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def fresnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "fresnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_fresnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_fresnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_fresnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_fresnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )
