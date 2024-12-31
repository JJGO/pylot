from pylot.util import to_device
import torch
from pylot.experiment import TrainExperiment


class HalfTE(TrainExperiment):
    def build_loss(self):
        super().build_loss()
        assert torch.cuda.is_available()
        # Creates once at the beginning of training
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def run_step(self, batch_idx, batch, backward=True, augmentation=True, epoch=None):

        x, y = to_device(
            batch, self.device, self.config.get("train.channels_last", False)
        )
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)

        if backward:
            # Scales the loss, and calls backward()
            # to create scaled gradients
            self.grad_scaler.scale(loss).backward()

            # Unscales gradients and calls
            # or skips optimizer.step()
            self.grad_scaler.step(self.optim)

            # Updates the scale for next iteration
            self.grad_scaler.update()

            self.optim.zero_grad()

        return {"loss": loss, "ytrue": y, "ypred": yhat}
