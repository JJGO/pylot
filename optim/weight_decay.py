from torch.nn.modules.batchnorm import _BatchNorm



def weight_decay_groups(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    norm_params = []

    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            norm_params.append(module.weight)
            norm_params.append(module.bias)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(param is x for x in norm_params):
            no_decay.append(param)
        elif "bias" in name:
            no_decay.append(param)
        elif  name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
