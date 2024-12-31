# modified from  
# https://github.com/ludvb/batchrenorm/blob/master/batchrenorm/batchrenorm.py
# https://github.com/mf1024/Batch-Renormalization-PyTorch/blob/master/batch_renormalization.py
import torch
from pydantic import validate_arguments
import einops as E


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.jit.ScriptModule):
    @validate_arguments
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        max_inc_step: float = 1e-4,
        max_r_max: float = 3.0,
        max_d_max: float = 5.0,
    ):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        self.max_inc_step = max_inc_step
        self.max_r_max = max_r_max
        self.max_d_max = max_d_max

        # Buffers
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer("running_std", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Params
        if self.affine:
            self.weight = torch.nn.Parameter(
                torch.ones(num_features, dtype=torch.float)
            )
            self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))

        # max multiplicative change
        self.register_buffer("r_max", torch.tensor(1.0, dtype=torch.float))
        # max additive change
        self.register_buffer("d_max", torch.tensor(0.0, dtype=torch.float))

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        dims = (0, *range(2, x.dim()))
        shape = (1, self.num_features, *[1 for _ in range(2, x.dim())])
        if self.training:
            batch_mean = x.mean(dim=dims)
            batch_std = x.std(dim=dims, unbiased=False)
            batch_std = batch_std.clamp_min(self.eps)

            r = batch_std / self.running_std
            r = r.clamp(1 / self.r_max, self.r_max).detach()

            d = (batch_mean - self.running_mean) / self.running_std
            d = d.clamp(-self.d_max, self.d_max).detach()

            x = (x - batch_mean.view(shape)) / batch_std.view(shape)
            x = x * r.view(shape) + d.view(shape)

            # update running stats
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (batch_std.detach() - self.running_std)
            self.num_batches_tracked += 1

            # increase max_d,r range
            batch_size = x.shape[0]
            if self.r_max < self.max_r_max:
                self.r_max += batch_size * self.max_inc_step
            if self.d_max < self.max_d_max:
                self.d_max += batch_size * self.max_inc_step

        else:
            x = (x - self.running_mean.view(shape)) / self.running_std.view(shape)

        if self.affine:
            x = self.weight.view(shape) * x + self.bias.view(shape)

        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError(f"expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"expected 5D input (got {x.dim()}D input)")
