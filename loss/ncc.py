import math

import numpy as np
import torch
from torch import nn

from .util import _loss_module_from_func


def ncc_loss(y_pred, y_true, window=9, eps=1e-5):

    # In Pytorch dims are (Batch, Channels, *rest)
    dims = y_pred.size()[2:]
    ndims = len(dims)

    # get conv function
    conv_fn = getattr(nn.functional, f"conv{ndims}d")

    # Expand window and check size
    if isinstance(window, int):
        window = [window] * ndims
    else:
        window = window
    assert len(window) == ndims

    padding = tuple(math.floor(w / 2) for w in window)
    sum_filt = torch.ones([1, 1, *window]).to(y_pred.device)

    def local_sum(x):
        # TODO padding_mode seems broken in nn.functional
        return conv_fn(x, sum_filt, stride=1, padding=padding)

    # Compute intermediate CC terms
    I, J = y_pred, y_true
    I2 = I * I
    J2 = J * J
    IJ = I * J

    # compute local sums via convolution
    I_sum = local_sum(I)
    J_sum = local_sum(J)
    I2_sum = local_sum(I2)
    J2_sum = local_sum(J2)
    IJ_sum = local_sum(IJ)
    # compute cross correlation
    win_size = np.prod(window)

    cross = IJ_sum - I_sum * J_sum / win_size
    I_var = I2_sum - I_sum * I_sum / win_size
    J_var = J2_sum - J_sum * J_sum / win_size

    cc = cross * cross / (I_var * J_var + eps)

    # return negative cc.
    return -cc.mean()


NCCLoss = _loss_module_from_func("NCCLoss", ncc_loss)
