import torch

def total_variation2d(x):
    batch_size, C, H, W = x.size()

    dh = torch.abs(x[..., :-1, :-1] - x[..., 1:, :-1])
    dw = torch.abs(x[..., :-1, :-1] - x[..., :-1, 1:])
    dh = dh.view(dh.size(0), -1)
    dw = dw.view(dw.size(0), -1)
    tv = torch.sum(dh, axis=-1) + torch.sum(dw, axis=-1)
    return tv