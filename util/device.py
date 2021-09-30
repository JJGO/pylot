import torch


def to_device(inputs, device):

    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    if isinstance(inputs, tuple):
        return tuple(to_device(x, device) for x in inputs)
    if isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    raise TypeError(f"Type {type(inputs)} not supported")
