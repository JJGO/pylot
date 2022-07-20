import copy
from collections import OrderedDict
from functools import lru_cache

from torch import nn
from ...util.mapping import allbut

from .void import VoidModule
from .xparam import ExternalParameter


def _init_from_regular(self, module: nn.Module):
    self._external_params = OrderedDict()
    self.__dict__.update(allbut(module.__dict__, ["_parameters"]))
    self._parameters = {}
    for name, value in module._parameters.items():
        if isinstance(value, nn.Parameter):
            setattr(self, name, ExternalParameter(value.shape))
        else:
            setattr(self, name, value)


@lru_cache(maxsize=None)
def _voided_class(module_type) -> type:

    return type(
        f"Void{module_type.__name__}",
        (module_type, VoidModule),
        {
            "__init__": _init_from_regular,
            "extra_repr": lambda x: "",
        },
    )


def _voidify(module: nn.Module, recurse=True, memo=None, whitelist=None):

    # Skip non parametric modules
    if not any(True for _ in module.parameters()):
        return module

    if module not in memo:
        if len(memo) == 0:
            memo[module] = _voided_class(module.__class__)(module)
        # elif len(module._parameters) == 0:
        #     memo[module] = module
        elif whitelist is None or isinstance(module, whitelist):
            memo[module] = _voided_class(module.__class__)(module)
        else:
            memo[module] = copy.deepcopy(module)

    voided_module = memo[module]

    if recurse:
        for name, submodule in module.named_children():
            setattr(
                voided_module,
                name,
                _voidify(submodule, memo=memo, whitelist=whitelist),
            )
    return voided_module


def voidify(module, recurse=True, whitelist=None):
    memo = {}
    whitelist = tuple(whitelist) if whitelist is not None else None
    return _voidify(module, recurse=recurse, whitelist=whitelist, memo=memo)
