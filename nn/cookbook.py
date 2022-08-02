from collections import OrderedDict

from torch import nn

from ..util.future import remove_prefix


def freeze_batchnorm(model: nn.Module, freeze_affine=True, freeze_running=True):
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


def remove_state_dict_wrapper(state_dict: OrderedDict) -> OrderedDict:
    # Helper to deal with DataParallel / DistributedDataParallel annoyances
    if all(k.startswith("module.") for k in state_dict):
        return OrderedDict(
            {remove_prefix(k, "module."): v for k, v in state_dict.items()}
        )
    return state_dict


def auto_load_state_dict(module: nn.Module, state_dict: OrderedDict):
    module.load_state_dict(remove_state_dict_wrapper(state_dict))
