from functools import wraps
from torch import nn
import torch.nn.functional as F


def flatten_loss(loss_func, keep_channels=False):
    if keep_channels:

        @wraps(loss_func)
        def flattened_loss(yhat, y):
            batch_size, n_channels, *_ = y.shape
            return loss_func(
                yhat.view(batch_size, n_channels, -1),
                y.view(batch_size, n_channels, -1),
            )

    else:

        @wraps(loss_func)
        def flattened_loss(yhat, y):
            batch_size = y.size()[0]
            return loss_func(yhat.view(batch_size, -1), y.view(batch_size, -1))

    return flattened_loss


def flatten_loss_module(loss_module, keep_channels=False):
    if keep_channels:

        def forward(self, yhat, y):
            batch_size, n_channels, *_ = y.shape
            return loss_module.forward(
                self,
                yhat.view(batch_size, n_channels, -1),
                y.view(batch_size, n_channels, -1),
            )

    else:

        def forward(self, yhat, y):
            batch_size = y.size()[0]
            return loss_module.forward(
                self, yhat.view(batch_size, -1), y.view(batch_size, -1)
            )

    name = f"Flat{loss_module.__name__}"
    return type(name, (loss_module,), dict(forward=forward))


FlatMSELoss = flatten_loss_module(nn.MSELoss)
flat_mse_loss = flatten_loss(F.mse_loss)
