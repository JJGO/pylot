import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):

    """An actual sane implementation of frozen batch norm that
    entirely removes the possibility of either the parameters or
    the running stats changing
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        eps: float,
    ):
        super().__init__()
        assert isinstance(weight, torch.Tensor)
        assert isinstance(bias, torch.Tensor)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)
        self.eps = eps

    @property
    def num_features(self) -> int:
        return self.weight.shape[0]

    def forward(self, input: torch.Tensor):
        return torch.batch_norm(
            input=input,
            weight=self.weight,
            bias=self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=False,
            eps=self.eps,
            momentum=0,
            cudnn_enabled=torch.backends.cudnn.enabled,
        )

    @classmethod
    def fromBatchNorm2d(cls, module: nn.BatchNorm2d) -> "FrozenBatchNorm2d":
        assert isinstance(module, nn.BatchNorm2d)
        return cls(
            module.weight,
            module.bias,
            module.running_mean,
            module.running_var,
            module.eps,
        )

    @classmethod
    def patch_module(cls, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                patched = cls.fromBatchNorm2d(child)
                setattr(module, name, patched)
            else:
                cls.patch_module(child)

    @classmethod
    def toBatchNorm2d(cls, module: "FrozenBatchNorm2d") -> nn.BatchNorm2d:
        assert isinstance(module, FrozenBatchNorm2d)
        module = nn.BatchNorm2d(num_features)

    @classmethod
    def unpatch_module(cls, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, FrozenBatchNorm2d):
                bn = nn.BatchNorm2d(child.num_features)
                bn.weight.data = child.weight.data
                bn.bias.data = child.bias.data
                bn.running_mean.data = child.running_mean.data
                bn.running_var.data = child.running_var.data
            else:
                cls.unpatch_module(child)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_features={self.num_features})"


def remove_batchnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            remove_batchnorm(child)


if __name__ == "__main__":

    import copy
    from typing import Tuple
    from tqdm.auto import tqdm
    from torchvision.models import resnet50

    def test_equal(
        module: nn.Module, input_shape: Tuple[int, ...], num_iter: int = 100
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module.eval()
        module = module.to(device)
        frozenbn_module = copy.deepcopy(module)
        FrozenBatchNorm2d.patch_module(frozenbn_module)

        for _ in tqdm(range(num_iter), leave=False):
            x = torch.randn(input_shape, device=device)
            assert torch.allclose(module(x), frozenbn_module(x))

    imagenet_shape = (10, 3, 224, 224)
    basic_model = nn.BatchNorm2d(3)
    compound_model = nn.Sequential(*[nn.BatchNorm2d(3) for _ in range(10)])
    imagenet_clf = resnet50(pretrained=True)
    test_equal(basic_model, imagenet_shape)
    test_equal(compound_model, imagenet_shape)
    test_equal(imagenet_clf, imagenet_shape)
