import torch
from torch import nn

# See:
#
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
# https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/


class SqueezeExcite2d(nn.Module):
    """Squeeze-and-Excitation block from (`Hu et al, 2019 <https://arxiv.org/abs/1709.01507>`_)

    This block applies global average pooling to the input, feeds the resulting
    vector to a single-hidden-layer fully-connected network (MLP), and uses the
    outputs of this MLP as attention coefficients to rescale the input. This
    allows the network to take into account global information about each input,
    as opposed to only local receptive fields like in a convolutional layer.

    Args:
        num_features (int): Number of features or channels in the input.
        latent_channels (float, optional): Dimensionality of the hidden layer within the added
            MLP. If less than 1, interpreted as a fraction of ``num_features``. Default: ``0.125``.
    """

    def __init__(self, num_features: int, latent_channels: float = 0.125):
        super().__init__()
        self.latent_channels = int(
            latent_channels if latent_channels >= 1 else latent_channels * num_features
        )
        flattened_dims = num_features

        self.pool_and_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(flattened_dims, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, num_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = input.shape
        attention_coeffs = self.pool_and_mlp(input)
        return input * attention_coeffs.reshape(n, c, 1, 1)


# see: https://github.com/qubvel/segmentation_models.pytorch/blob/8523324c116dcf7be6bddb73bf4eb1779ef6e611/segmentation_models_pytorch/base/modules.py#L84
class SCSE2d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
