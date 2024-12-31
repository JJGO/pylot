import torch

from pylot.util import autosave, autoload

from ..nn.ema import ModelEMA


class EMA:
    def __init__(self, experiment, **kwargs):
        self.exp = experiment
        self.ema = ModelEMA(self.exp.model, **kwargs)
        self.last_epoch = self.exp.properties.get("epoch", None)

        self._ckpt_path = self.exp.path / "checkpoints/ema.pt"
        if self.last_epoch is not None:
            self.load_ema()

    def load_ema(self):
        state = autoload(self._ckpt_path)
        self.ema.module.load_state_dict(state["model"])

    def save_ema(self):
        state = {
            "model": self.ema.module.state_dict(),
            "_epoch": self.last_epoch,
        }
        autosave(state, self._ckpt_path)

    def __call__(self, phase, epoch, **kwargs):
        if phase == "train":
            if self.last_epoch is None:
                self.last_epoch = epoch

            if epoch > self.last_epoch:
                self.last_epoch = epoch
                self.save_ema()

            self.ema.update(self.exp.model)
