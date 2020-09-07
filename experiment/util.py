import random

import torch
import numpy as np


def fix_seed(seed=42, deterministic=False):
    # https://pytorch.org/docs/stable/notes/randomness.html

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def any_getattr(modules, attr):
    for module in reversed(modules):
        if hasattr(module, attr):
            return getattr(module, attr)
    module_names = [m.__name__ for m in modules]
    raise ImportError(f"Attribute {attr} not found in any of {module_names}")
