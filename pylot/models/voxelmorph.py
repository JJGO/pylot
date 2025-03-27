from typing import Dict, Tuple, Any

import torch
from torch import nn, Tensor

from ..nn.spatial_transformer import SpatialTransformer
from .unet import UNet


class Voxelmorph(nn.Module):
    def __init__(
        self,
        size: Tuple[int, ...],
        unet: Dict[str, Any],
        in_channels=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        dims = len(size)
        self.registration_unet = UNet(
            in_channels=in_channels, out_channels=dims, **unet
        )
        self.spatial_transform = SpatialTransformer(size)

    def forward(self, Xm: Tensor, Xf: Tensor) -> Dict[str, Tensor]:

        phi = self.registration_unet(torch.cat([Xm, Xf], dim=1))

        Xm_phi = self.spatial_transform(Xm, phi)

        return dict(Xm_phi=Xm_phi, phi=phi)
