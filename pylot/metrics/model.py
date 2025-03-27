import pandas as pd


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
