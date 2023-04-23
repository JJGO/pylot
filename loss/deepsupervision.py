from typing import Callable, Optional, Sequence

from torch import Tensor, nn

from ..nn import resize


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss_func: Callable, weights: Optional[Sequence] = None):
        super().__init__()
        self.loss_func = loss_func
        self.weights = weights

    # TODO only works for 2d right now

    def forward(self, y_preds: Sequence[Tensor], y_true: Tensor):
        weights = self.weights or [1] * len(y_preds)
        assert len(weights) == len(y_preds)

        total_loss = 0
        for y_pred, weight in zip(y_preds, weights):
            if y_pred.shape != y_true.shape:
                y_pred = resize(y_pred, size=y_true.shape[-2:])
            total_loss = weight * self.loss_func(y_pred, y_true)
        return total_loss
