from torch import nn


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
