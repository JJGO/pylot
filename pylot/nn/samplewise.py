from typing import List

import numpy as np
import torch
from torch import Tensor
from torch import nn


class SamplewiseLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, xs: List[Tensor]) -> List[Tensor]:
        norm = []
        for x in xs:
            mu = x.mean()
            var = x.var(unbiased=False)
            norm.append((x - mu) / torch.sqrt(var + self.eps))

        if not self.elementwise_affine:
            return norm

        return [x * self.weight + self.bias for x in norm]


def test_samplewiselayernorm(
    n_iters=100, size=(3, 32, 32), batch_size=10, atol=1e-5, elementwise_affine=True
):

    for _ in range(n_iters):
        data = np.array(
            [np.random.normal(i + 1, i + 1, size=size) for i in range(batch_size)]
        ).astype(np.float32)

        x = torch.tensor(data.copy(), requires_grad=True)

        ref = nn.LayerNorm(size, elementwise_affine=elementwise_affine)
        y = ref(x)
        y.sum().backward()

        xs = [torch.tensor(d.copy(), requires_grad=True) for d in data]
        mine = SamplewiseLayerNorm(size, elementwise_affine=elementwise_affine)
        ys = mine(xs)
        torch.stack(ys).sum().backward()

        assert torch.allclose(y, torch.stack(ys), atol=atol)
        if elementwise_affine:
            assert torch.allclose(ref.weight.grad, mine.weight.grad, atol=atol)
            assert torch.allclose(ref.bias.grad, mine.bias.grad, atol=atol)


class SamplewiseBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        equal_weighting=True,
    ):
        # By default, all samples contribute equally regardless of the spatial dimensions
        # I.E. given samples of sizes (C, H, W) and (C, H', W') both contribute equally to mean and variance
        # despite the difference in the remainder dimensions. equal_weighthing=False weights them per pixel
        # instead of per sample
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.equal_weighting = equal_weighting

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, xs: List[Tensor]) -> List[Tensor]:
        axes = list(range(1, xs[0].dim()))
        if self.training:
            mu = torch.zeros(self.num_features, device=xs[0].device)
            var = torch.zeros(self.num_features, device=xs[0].device)
            if self.equal_weighting:
                # Welford rules
                # mu = 1/m \sum_i mu_i
                # var = -mu^2 + 1/m \sum_i var_i + mu_i^2
                mu_i = torch.stack([x.mean(axes) for x in xs])
                var_i = torch.stack([x.var(axes, unbiased=False) for x in xs])
                mu = mu_i.mean(dim=0)
                var = (var_i + mu_i * mu_i).mean(dim=0) - mu * mu
            else:
                mu = torch.zeros(self.num_features, device=xs[0].device)
                var = torch.zeros(self.num_features, device=xs[0].device)
                n = 0
                for x in xs:
                    n_i = np.prod(x.shape[1:])
                    n += n_i
                    mu_i = x.mean(axes)
                    mu += n_i * mu_i
                    var += n_i * (x.var(axes, unbiased=False) + mu_i ** 2)
                mu /= n
                var = var / n - mu ** 2

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    self.running_mean = (
                        self.running_mean * (1 - self.momentum) + mu * self.momentum
                    )
                    self.running_var = (
                        self.running_var * (1 - self.momentum) + var * self.momentum
                    )
        else:
            mu = self.running_mean
            var = self.running_var

        norm = [
            (x - mu[:, None, None]) / torch.sqrt(var[:, None, None] + self.eps)
            for x in xs
        ]

        if not self.affine:
            return norm

        return [self.weight[:, None, None] * x + self.bias[:, None, None] for x in norm]


def test_samplewisebatchrnorm(
    n_iters=100, size=(3, 32, 32), batch_size=10, atol=1e-5, affine=True
):

    for _ in range(n_iters):
        data = np.array(
            [np.random.normal(i + 1, i + 1, size=size) for i in range(batch_size)]
        ).astype(np.float32)

        x = torch.tensor(data.copy(), requires_grad=True)

        ref = nn.BatchNorm2d(size[0], affine=affine)
        y = ref(x)
        y.sum().backward()

        xs = [torch.tensor(d.copy(), requires_grad=True) for d in data]
        mine = SamplewiseBatchNorm(size[0], affine=affine, equal_weighting=False)
        ys = mine(xs)
        torch.stack(ys).sum().backward()

        assert torch.allclose(y, torch.stack(ys), atol=atol)
        if affine:
            assert torch.allclose(ref.weight.grad, mine.weight.grad, atol=atol)
            assert torch.allclose(ref.bias.grad, mine.bias.grad, atol=atol)


if __name__ == "__main__":
    test_samplewiselayernorm()
    test_samplewisebatchrnorm(affine=False)
