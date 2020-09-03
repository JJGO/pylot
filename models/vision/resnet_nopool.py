from torch import nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    ResNet,
)


def remove_first_maxpool(resnet):
    assert isinstance(resnet, ResNet)
    resnet.maxpool = nn.Identity()
    return resnet


def resnet18_nopool(**kwargs):
    m = resnet18(**kwargs)
    return remove_first_maxpool(m)

def resnet34_nopool(**kwargs):
    m = resnet34(**kwargs)
    return remove_first_maxpool(m)


def resnet50_nopool(**kwargs):
    m = resnet50(**kwargs)
    return remove_first_maxpool(m)


def resnet101_nopool(**kwargs):
    m = resnet101(**kwargs)
    return remove_first_maxpool(m)


def resnet152_nopool(**kwargs):
    m = resnet152(**kwargs)
    return remove_first_maxpool(m)
