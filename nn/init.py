import torch
from torch import nn
from torch.nn import init


def initialize_tensor(tensor, distribution, nonlinearity="LeakyReLU"):

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution is None:
        pass
    elif distribution == "zeros":
        init.zeros_(tensor)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(tensor, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(tensor, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(tensor, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(tensor, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(tensor, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(tensor, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(tensor, gain)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'")


def initialize_layer(layer, distribution=None, zero_bias=True, nonlinearity=None):

    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), "Can only be applied to linear and conv layers"

    initialize_tensor(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        if zero_bias:
            initialize_tensor(layer.bias, "zeros", nonlinearity)
        else:
            initialize_tensor(layer.bias, distribution, nonlinearity)
