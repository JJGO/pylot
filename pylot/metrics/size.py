import numpy as np
import torch

# https://pytorch.org/docs/stable/tensor_attributes.html
dtype2bits = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def model_size(model, as_bits=False):
    """Returns model size in #params or #bits

    Arguments:
        model {torch.nn.Module} -- Network to compute model size over

    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype

    Returns:
        int -- Total number of weight & bias params
    """

    total_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            t *= bits
        total_params += t
    return int(total_params)
