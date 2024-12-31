import pathlib
from typing import Optional

import torch
from .train import TrainExperiment
from ..util import Config, autoload


class FinetuningExperiment(TrainExperiment):
    
    # TODO: Add batch norm freeze/fuse/finetune support

    @property
    def parent(self):
        parent = self.config["finetune.parent"]
        return pathlib.Path(parent)

    def build_model(self):
        super().build_model()
        checkpoint = self.config["finetune.checkpoint"]
        initial_checkpoint = self.parent / f"checkpoints/{checkpoint}.pt"
        if initial_checkpoint.exists():
            model_weights = torch.load(initial_checkpoint)["model"]
            self.model.load_state_dict(model_weights)
        else:
            print(
                f"FAILED to load initial checkpoint because it could not be found: {str(initial_checkpoint)}"
            )

    @classmethod
    def from_other(cls, path: pathlib.Path, cfg_update: Optional[Config]):
        assert "finetune" in cfg_update
        base_cfg = Config(autoload(path / "config.yml"))
        cfg = base_cfg.update(cfg_update)
        cfg = cfg.set("finetune.parent", str(path))
        return cls.from_config(cfg)
