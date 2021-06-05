from torch import nn

def num_params(module: nn.Module, only_learnable=True):
    total = 0
    for p in module.parameters():
        if not only_learnable or p.requires_grad:
            total += p.numel()
    return total

