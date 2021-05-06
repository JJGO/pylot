from torch import Tensor

def batch_channel_flatten(x: Tensor) -> Tensor:
    batch_size, n_channels, *_ = x.shape
    return x.view(batch_size, n_channels, -1)
