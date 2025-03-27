from collections import OrderedDict
from typing import Literal

from torch import nn
import pandas as pd

from .hooks import HookedModule
from ..util.future import remove_prefix


def freeze_batchnorm(
    model: nn.Module, freeze_affine=True, freeze_running=True, stats="running"
):

    bn_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, bn_classes):
            # Freeze the affine params
            if hasattr(module, "weight"):
                module.weight.requires_grad_(not freeze_affine)
            if hasattr(module, "bias"):
                module.bias.requires_grad_(not freeze_affine)

            # Freeze running stats
            module.track_running_stats = not freeze_running
            module.eval()


def set_batchnorm(
    model: nn.Module,
    learn_affine=True,
    track_running=True,
    stats: Literal["running", "batch"] = "batch",
):

    assert stats in ("batch", "running"), 'stats must be one of ["running", "batch"]'

    bn_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, bn_classes):
            # Freeze the affine params
            if hasattr(module, "weight"):
                module.weight.requires_grad_(learn_affine)
            if hasattr(module, "bias"):
                module.bias.requires_grad_(learn_affine)

            # Freeze running stats
            module.track_running_stats = track_running

            if stats == "running":
                module.eval()
            elif stats == "batch":
                module.train()


def split_param_groups_by_weight_decay(model: nn.Module, weight_decay: int = 0):
    if weight_decay == 0:
        return [{"params": model.parameters(), "weight_decay": 0}]

    nonWD_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)

    WD_list = []
    nonWD_list = []
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if isinstance(module, nonWD_modules):
                nonWD_list.append(param)
            else:
                WD_list.append(param)
    return [
        {"params": WD_list, "weight_decay": weight_decay},
        {"params": nonWD_list, "weight_decay": 0},
    ]


def filter_state_dict(state_dict: OrderedDict, prefix) -> OrderedDict:
    # Helper to remove a given prefix from a state_dict
    # common
    return OrderedDict(
        {
            remove_prefix(k, f"{prefix}."): v
            for k, v in state_dict.items()
            if k.startswith(f"{prefix}.")
        }
    )


def remove_state_dict_wrapper(state_dict: OrderedDict) -> OrderedDict:
    # Helper to deal with DataParallel / DistributedDataParallel annoyances
    if all(k.startswith("module.") for k in state_dict):
        return filter_state_dict(state_dict, "module")
    return state_dict


def auto_load_state_dict(module: nn.Module, state_dict: OrderedDict):
    module.load_state_dict(remove_state_dict_wrapper(state_dict))


def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.

    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.

    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = False


def activate_requires_grad(model: nn.Module):
    """Activates the requires_grad flag for all parameters of a model.

    Use this method to activate gradients for a model (e.g. after deactivating
    them using `deactivate_requires_grad(...)`).

    Examples:
        >>> backbone = resnet18()
        >>> activate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = True


def num_params(module: nn.Module, only_learnable=True):
    total = 0
    for p in module.parameters():
        if not only_learnable or p.requires_grad:
            total += p.numel()
    return total


def parameter_table(model):

    rows = []

    module_dict = {k: m.__class__.__name__ for k, m in model.named_modules()}

    for name, tensor in model.named_parameters():

        if "." in name:
            module = module_dict[name[: name.rindex(".")]]
        else:
            module = model.__class__.__name__

        rows.append(
            dict(
                module=module,
                param=name,
                shape=tuple(tensor.size()),
                numel=tensor.numel(),
                grad=tensor.requires_grad,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        )

    df = pd.DataFrame.from_records(rows)
    df["percent"] = (df.numel / df.numel.sum() * 100).round(3)

    return df


def module_table(model):

    rows = []

    for name, module in model.named_modules():

        rows.append(dict(name=name, module=module.__class__.__name__))

    return pd.DataFrame.from_records(rows)


def trace_shapes(model, *inputs, module_types=None, glob=None):
    module_names = {m: k for k, m in model.named_modules()}
    rows = []

    def trace(module: nn.Module, inputs, output):
        name = module_names[module]
        input_shapes = [tuple(i.shape) for i in inputs]
        if isinstance(output, tuple):
            output_shape = [tuple(i.shape) for i in output]
        else:
            output_shape = tuple(output.shape)

        entry = dict(
            name=name,
            module=module.__class__.__name__,
            input_shapes=input_shapes,
            output_shape=output_shape,
        )
        rows.append(entry)

    with HookedModule(
        model,
        trace,
        module_types=module_types,
        glob=glob,
    ):
        model(*inputs)

    return pd.DataFrame.from_records(rows)
