from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is None:
            weights = [1 for _ in losses]
        assert len(weights) == len(losses)
        self.weights = weights
        self.losses = losses

    def forward(self, pred, target):
        return sum(w * fn(pred, target) for w, fn in zip(self.weights, self.losses))
