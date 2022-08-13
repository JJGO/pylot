import copy
import pathlib
import sys
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseExperiment
from .util import eval_config, absolute_import
from ..util.torchutils import to_device
from ..util.meter import MeterDict
from ..util.ioutil import autosave
from ..nn.util import num_params, split_param_groups_by_weight_decay


class TrainExperiment(BaseExperiment):
    def __init__(self, path):
        torch.backends.cudnn.benchmark = True
        super().__init__(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_data()
        self.build_model()
        self.build_optim()
        self.build_loss()
        self.build_metrics()
        self.build_augmentations()

    def build_data(self):
        data_cfg = self.config["data"].to_dict()
        dataset_constructor = data_cfg.pop('_class', None) or data_cfg.pop('_fn')
        dataset_cls = absolute_import(dataset_constructor)

        self.train_dataset = dataset_cls(split="train", **data_cfg)
        self.val_dataset = dataset_cls(split="val", **data_cfg)

    def build_dataloader(self):
        dl_cfg = self.config["dataloader"]
        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, drop_last=True, **dl_cfg
        )
        self.val_dl = DataLoader(
            self.val_dataset, shuffle=False, drop_last=False, **dl_cfg
        )

    def build_model(self):
        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)

    def build_optim(self):
        optim_cfg = self.config["optim"].to_dict()
        # TODO add lr_scheduler support
        # self.lr_scheduler = eval_config(opt_cfg.pop('lr_scheduler', None)

        if "weight_decay" in optim_cfg:
            optim_cfg["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg["weight_decay"]
            )
        else:
            optim_cfg["params"] = self.model.parameters()

        self.optim = eval_config(optim_cfg)

    def build_loss(self):
        self.loss_func = eval_config(self.config["loss_func"])

    def build_metrics(self):
        self.metric_fns = {}
        if "log.metrics" in self.config:
            self.metric_fns = eval_config(copy.deepcopy(self.config["log.metrics"]))

    def build_initialization(self):
        if "initialization" in self.config:
            init_cfg = self.config["initialization"].to_dict()
            path = pathlib.Path(init_cfg["path"])
            with path.open("rb") as f:
                state = torch.load(f, map_location=self.device)
            if not init_cfg.get("optim", True):
                state.pop("optim", None)
            strict = init_cfg.get("strict", True)
            self.set_state(state, strict=strict)
            print(f"Loaded initialization state from: {path}")

    @property
    def state(self):
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "_epoch": self.properties["epoch"],
        }

    def set_state(self, state, strict=True):
        for attr, state_dict in state.items():
            if not attr.startswith("_"):
                x = getattr(self, attr)
                if isinstance(x, nn.Module):
                    x.load_state_dict(state_dict, strict=strict)
                elif isinstance(x, torch.optim.Optimizer):
                    x.load_state_dict(state_dict)
                else:
                    raise TypeError(f"Unsupported type {type(x)}")

    def checkpoint(self, tag=None):
        self.properties["epoch"] = self._epoch

        checkpoint_dir = self.path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        tag = tag if tag is not None else "last"
        print(f"Checkpointing with tag:{tag} at epoch:{self._epoch}")

        with (checkpoint_dir / f"{tag}.pt").open("wb") as f:
            torch.save(self.state, f)

    @property
    def checkpoints(self, as_paths=False) -> List[str]:
        checkpoints = list((self.path / "checkpoints").iterdir())
        checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
        if as_paths:
            return checkpoints
        return [c.stem for c in checkpoints]

    def load(self, tag=None):
        checkpoint_dir = self.path / "checkpoints"
        tag = tag if tag is not None else "last"
        with (checkpoint_dir / f"{tag}.pt").open("rb") as f:
            state = torch.load(f, map_location=self.device)
            self.set_state(state)
        print(
            f"Loaded checkpoint with tag:{tag}. Last epoch:{self.properties['epoch']}"
        )
        return self

    def to_device(self):
        self.model = to_device(
            self.model, self.device, self.config.get("train.channels_last", False)
        )

    def run_callbacks(self, callback_group, **kwargs):
        for callback in self.callbacks.get(callback_group, []):
            callback(**kwargs)

    def run(self):
        self.build_dataloader()
        self.build_callbacks()
        self.to_device()

        print(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]

        last_epoch: int = self.properties.get("epoch", -1)
        if last_epoch >= 0:
            self.load(tag="last")
            df = self.metrics.df
            autosave(df[df.epoch < last_epoch], self.path / "metrics.jsonl")
        else:
            self.build_initialization()

        self.to_device()
        self.optim.zero_grad()

        checkpoint_freq: int = self.config.get("log.checkpoint_freq", 1)
        eval_freq: int = self.config.get("train.eval_freq", 1)

        self.run_callbacks("setup")

        # try:

        for epoch in range(last_epoch + 1, epochs):
            print(f"Start epoch {epoch}")
            self._epoch = epoch
            self.run_phase("train", epoch)
            if epoch % eval_freq == 0 or epoch == epochs - 1:
                self.run_phase("val", epoch)

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.checkpoint()
            self.run_callbacks("epoch", epoch=epoch)

        self.checkpoint(tag="last")

        self.run_callbacks("wrapup")

        # except KeyboardInterrupt:
        #     print(f"Interrupted at epoch {epoch}. Tearing Down")
        #     self.checkpoint(tag="interrupt")
        #     sys.exit(1)

    def run_phase(self, phase, epoch):

        dl = getattr(self, f"{phase}_dl")

        grad_enabled = phase == "train"
        augmentation = (phase == "train") and ("augmentations" in self.config)

        self.model.train(grad_enabled)  # For dropout, batchnorm, &c

        meters = MeterDict()

        with torch.set_grad_enabled(grad_enabled):
            # with torch.inference_mode(not grad_enabled):
            for batch_idx, batch in enumerate(dl):
                outputs = self.run_step(
                    batch_idx,
                    batch,
                    backward=grad_enabled,
                    augmentation=augmentation,
                    epoch=epoch,
                )
                metrics = self.compute_metrics(outputs)
                meters.update(metrics)
                self.run_callbacks("batch", epoch=epoch, batch_idx=batch_idx)

        metrics = {"phase": phase, "epoch": epoch, **meters.collect("mean")}
        self.metrics.log(metrics)
        return metrics

    def run_step(self, batch_idx, batch, backward=True, augmentation=True, epoch=None):

        x, y = to_device(
            batch, self.device, self.config.get("train.channels_last", False)
        )

        if augmentation:
            with torch.no_grad():
                x = self.aug_pipeline(x)

        yhat = self.model(x)
        loss = self.loss_func(yhat, y)

        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return {"loss": loss, "ytrue": y, "ypred": yhat}

    def compute_metrics(self, outputs):
        metrics = {"loss": outputs["loss"].item()}
        for name, fn in self.metric_fns.items():
            value = fn(outputs["ypred"], outputs["ytrue"])
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics[name] = value
        return metrics

    def build_augmentations(self):
        # if "augmentations" in self.config:
        #     aug_cfg = self.config.to_dict()["augmentations"]
        #     self.aug_pipeline = augmentations_from_config(aug_cfg)
        pass


# Example use:
# >>> from universeg.experiment import TrainExperiment
# >>> exp =  TrainExperiment.from_config(config)
# >>> exp.run()
