import pandas as pd
from torch import nn

from .hooks import HookedModule


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
        model, trace, module_types=module_types, glob=glob,
    ):
        model(*inputs)

    return pd.DataFrame.from_records(rows)
