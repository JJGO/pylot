from contextlib import contextmanager
import fnmatch

from torch import nn


@contextmanager
def HookedModule(
    module: nn.Module, hookfn, mode: str = "forward", module_types=None, glob=None
):
    assert mode in ("forward", "backward")

    hooks = []

    for name, mod in module.named_modules():
        if module_types and not isinstance(mod, module_types):
            continue
        if glob and not fnmatch.fnmatch(name, glob):
            continue
        if mode == "forward":
            hooks.append(mod.register_forward_hook(hookfn))
        else:
            hooks.append(mod.register_backward_hook(hookfn))
    try:
        yield module
    finally:
        for hook in hooks:
            hook.remove()
