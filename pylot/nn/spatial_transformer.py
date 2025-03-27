import torch
from torch import nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode="bilinear", order="image"):
        assert order in ("image", "tensor")
        super().__init__()

        self.size = size
        self.mode = mode
        self.order = order

        self._init_grid()
        self.register_buffer("_s", torch.Tensor(self.size).flip([-1]) - 1)

    def _init_grid(self):

        vectors = [torch.arange(0, s).type(torch.FloatTensor) for s in self.size]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij'), dim=-1)
        # grid_resample expects image order, i.e. (y,x) or (z,y,x)
        grid = grid.flip(dims=[-1])
        grid = grid.unsqueeze(0)
        self.register_buffer("grid", grid)

    def forward(self, src, flow):

        # Move channel axis to end
        flow = flow.moveaxis(1, -1)
        if self.order == "tensor":
            flow = flow.flip(dims=[-1])

        new_locs = self.grid + flow

        # Map values from [0,N] to [-1,1], where N is size of dimension
        new_locs = 2 * (new_locs / self._s - 0.5)

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

