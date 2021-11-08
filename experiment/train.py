import copy
from collections import defaultdict
import functools
import inspect
import json
import pathlib
import sys

from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.optim
from torch.optim import lr_scheduler

import torchvision.datasets
import torchvision.models


from .base import Experiment
from ..datasets import stratified_train_val_split
from ..loss import flatten_loss
from ..util import (
    printc,
    StatsMeter,
    StatsTimer,
    StatsCUDATimer,
    allbut,
    make_path,
    to_device,
    any_getattr,
)
from .. import callbacks
from .. import datasets
from .. import models
from .. import optim
from .. import loss as custom_loss
from .. import metrics
from ..optim import scheduler as custom_scheduler
from ..optim.scheduler import WarmupScheduler


class TrainExperiment(Experiment):

    MODELS = [torchvision.models, models]
    DATASETS = [torchvision.datasets, datasets]
    CALLBACKS = [callbacks]
    LOSS = [torch.nn, custom_loss]
    OPTIMS = [torch.optim, optim]
    SCHEDULERS = [lr_scheduler, custom_scheduler]
    METRICS = [metrics]

    def __init__(self, cfg=None, **kwargs):

        # Default children kwargs
        super().__init__(cfg, **kwargs)

        # Attributes
        self.train_dataset = None
        self.val_dataset = None
        self.train_dl = None
        self.val_dl = None
        self.model = None
        self.loss_func = None
        self.epochs = None
        self._epoch = None
        self.setup_callbacks = None
        self.batch_callbacks = None
        self.epoch_callbacks = None
        self.wrapup_callbacks = None
        self.metric_fns = None
        self.device = None
        self.checkpoint_freq = 1

        self.build_data(**self.cfg["data"])
        self.build_model(**self.cfg["model"])
        if "loss" in self.cfg:
            self.build_loss(**self.cfg["loss"])
        if "losses" in self.cfg:
            self.build_losses(**self.cfg["losses"])
        self.build_train(**self.cfg["train"])

    def build_data(self, dataset, val_split=None, **data_kwargs):

        constructor = any_getattr(self.DATASETS, dataset)
        kwargs = allbut(data_kwargs, ["dataloader"])
        self.dataset = constructor(train=True, **kwargs)
        self.test_dataset = constructor(train=False, **kwargs)
        if val_split is not None:
            # Replicates should be done for weight initialization, they should not
            # touch how data is partitioned, for that make use of a fold parameter
            # in the dataset and split using sklearn or something
            self.train_dataset, self.val_dataset = stratified_train_val_split(
                self.dataset, val_split, seed=data_kwargs.get("seed", 42)
            )
        else:
            self.train_dataset = self.dataset
            self.val_dataset = self.test_dataset

        self.build_dataloader(**data_kwargs["dataloader"])

    def build_dataloader(self, **dataloader_kwargs):

        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, **dataloader_kwargs
        )
        self.val_dl = DataLoader(self.val_dataset, shuffle=True, **dataloader_kwargs)
        self.test_dl = DataLoader(self.test_dataset, shuffle=False, **dataloader_kwargs)

    def build_model(self, model, weights=None, **model_kwargs):
        constructor = any_getattr(self.MODELS, model)
        self.model = constructor(**model_kwargs)

        if weights is not None:
            with make_path(weights).open("rb") as f:
                self.load_model(
                    torch.load(
                        f,
                        map_location=(
                            None if torch.cuda.is_available() else torch.device("cpu")
                        ),
                    )
                )

    def build_loss(self, loss_func=None, **loss_kwargs):
        if loss_func == "MultiLoss":
            losses = []
            for sub_loss in loss_kwargs["losses"]:
                sub_loss_func = sub_loss.pop("loss_func")
                losses.append(any_getattr(self.LOSS, sub_loss_func)(**sub_loss))
            loss_kwargs["losses"] = losses
        loss_func = any_getattr(self.LOSS, loss_func)(**loss_kwargs)

        self.loss_func = loss_func

    def build_losses(self, **losses):
        losses = copy.deepcopy(losses)
        self.losses = {}
        self.loss_weights = {}
        for name, loss_kwargs in losses.items():
            self.loss_weights[name] = loss_kwargs.pop("loss_weight", 1)
            loss_func = loss_kwargs.pop("loss_func")
            self.losses[name] = any_getattr(self.LOSS, loss_func)(**loss_kwargs)

    def build_train(
        self,
        optim,
        epochs,
        scheduler=None,
        warmup=None,
        accumulate_gradients=None,
        **train_kws,  # Extra parameters that are manually retrieved with self.get_param
    ):

        self.epochs = epochs
        self.accumulate_gradients = accumulate_gradients

        # Optim
        if isinstance(optim, dict):
            optim, optim_kwargs = optim["optim"], allbut(optim, ["optim", "state"])
            constructor = any_getattr(self.OPTIMS, optim)
            optim = constructor(self.model.parameters(), **optim_kwargs)
        self.optim = optim
        optim_state = self.get_param("train.optim.state", None)
        if optim_state:
            self.load_optim(optim_state)

        # Scheduler
        self.scheduler = scheduler
        if scheduler is not None:
            scheduler, scheduler_kwargs = (
                scheduler["scheduler"],
                allbut(scheduler, ["scheduler", "state"]),
            )
            constructor = any_getattr(self.SCHEDULERS, scheduler)
            self.scheduler = constructor(self.optim, **scheduler_kwargs)

        if warmup is not None:
            warmup_period = len(self.train_dl) * warmup
            # We jump the scheduler ahead so warmup reaches the correct point
            self.scheduler = WarmupScheduler(warmup_period, self.scheduler, skip=warmup)

        scheduler_state = self.get_param("train.scheduler.state", None)
        if scheduler_state:
            self.load_scheduler(scheduler_state)

    def _load_module(self, checkpoint, module, ignore_missing=False):
        # assert checkpoint.exists(), f"Checkpoint path {checkpoint} does not exist"
        # checkpoint = torch.load(checkpoint)
        checkpoint = self._path_to_checkpoint(checkpoint)
        if module == "model":
            getattr(self, module).load_state_dict(
                checkpoint[f"{module}_state_dict"], strict=not ignore_missing
            )
        else:
            getattr(self, module).load_state_dict(checkpoint[f"{module}_state_dict"])

    def load_model(self, checkpoint, ignore_missing=False):
        self._load_module(checkpoint, "model", ignore_missing=ignore_missing)

    def load_optim(self, checkpoint):
        self._load_module(checkpoint, "optim")

    def load_scheduler(self, checkpoint):
        self._load_module(checkpoint, "scheduler")

    def to_device(self, device=None):
        # Torch CUDA config
        if device:
            self.device = device
        elif self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="RED")
        self.model.to(self.device)
        if isinstance(self.loss_func, torch.nn.Module):
            self.loss_func.to(self.device)
        cudnn.benchmark = True  # For fast training.

    @property
    def checkpoint_path(self):
        return self.path / "checkpoints"

    def checkpoint(self, tag=None):
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

        tag = tag if tag is not None else "last"
        printc(f"Checkpointing with tag:{tag} at epoch:{self._epoch}", color="BLUE")
        checkpoint_file = f"{tag}.pt"
        state = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "epoch": self._epoch,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        with (self.checkpoint_path / checkpoint_file).open("wb") as f:
            torch.save(state, f)

    def load(self, tag=None):
        self.build_logging()
        self.to_device()
        # Load model & optimizer
        if not self.checkpoint_path.exists():
            printc("No checkpoints were found", color="ORANGE")
            self._epoch = 0
            return
        self.reload(tag=tag)

    def _path_to_checkpoint(self, path):
        if isinstance(path, pathlib.Path):
            with path.open("rb") as f:
                checkpoint = torch.load(
                    f,
                    map_location=(
                        None if torch.cuda.is_available() else torch.device("cpu")
                    ),
                )
            return checkpoint
        return path

    def reload(self, tag=None, ignore_missing=False):
        tag = tag if tag is not None else "last"
        checkpoint = self._path_to_checkpoint(self.checkpoint_path / f"{tag}.pt")
        self._epoch = checkpoint["epoch"]
        self.load_model(checkpoint, ignore_missing=ignore_missing)
        self.load_optim(checkpoint)
        if self.scheduler is not None:
            self.load_scheduler(checkpoint)
        printc(
            f"Loaded checkpoint with tag:{tag}. Last epoch:{self._epoch}", color="BLUE"
        )

    def build_metrics(self):

        self.metric_fns = {}

        for metric in self.get_param("log.metrics"):
            if isinstance(metric, str):
                f = any_getattr(self.METRICS, metric)
            elif isinstance(metric, dict):
                assert len(metric) == 1
                metric, kwargs = next(iter(metric.items()))
                f = any_getattr(self.METRICS, metric)
                if inspect.isfunction(f):
                    f = functools.partial(f, **kwargs)
                else:
                    f = f(**kwargs)
            else:
                raise TypeError(f"metric cannot be of type {type(metric)}")

            self.metric_fns[metric] = f

    def build_logging(self):
        super().build_logging()

        self.checkpoint_freq = self.get_param("log.checkpoint_freq", 1)

        # Callbacks
        if "log" in self.cfg:

            for category in ["setup", "batch", "epoch", "wrapup"]:
                category = f"{category}_callbacks"
                callbacks = []
                if category in self.cfg["log"]:

                    for cb in self.cfg["log"][category]:
                        if isinstance(cb, str):
                            k, args = cb, {}
                        else:
                            assert len(cb) == 1
                            k, args = next(iter(cb.items()))
                        callbacks.append(any_getattr(self.CALLBACKS, k)(self, **args))

                setattr(self, category, callbacks)

            if "metrics" in self.cfg["log"]:
                self.build_metrics()

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        try:
            for epoch in range(start, end):
                printc(f"Start epoch {epoch}", color="YELLOW")
                self._epoch = epoch
                if self.checkpoint_freq > 0 and epoch % self.checkpoint_freq == 0:
                    self.checkpoint(tag="last")
                self.train(epoch)
                self.eval(epoch)
                if self.scheduler:
                    self.scheduler.step()

                with torch.no_grad():
                    for cb in self.epoch_callbacks:
                        cb(epoch)

            self.test(end)

            for cb in self.wrapup_callbacks:
                cb()

            self.checkpoint(tag="last")

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color="RED")
            self.checkpoint(tag="interrupt")
            sys.exit(1)

    def run_epoch(self, phase, epoch=0):
        progress = self.get_param("log.progress", False)
        timing = self.get_param("log.timing", False)

        dl = getattr(self, f"{phase}_dl")

        grad_enabled = phase == "train"

        if grad_enabled:
            self.model.train()
        else:
            self.model.eval()

        meters = defaultdict(StatsMeter)
        timercls = StatsCUDATimer if torch.cuda.is_available() else StatsTimer
        timer = timercls(unit="ms", skip=10)
        if not timing:
            timer.disable()

        epoch_iter = iter(dl)
        if progress:
            epoch_progress = tqdm(epoch_iter)
            epoch_progress.set_description(
                f"{phase.capitalize()} Epoch {epoch}/{self.epochs}"
            )
            epoch_iter = iter(epoch_progress)

        self.before_epoch(phase, epoch)

        with torch.set_grad_enabled(grad_enabled):
            for batch_idx in range(len(dl)):
                self.before_batch(phase, epoch, batch_idx)
                outputs = self.run_step(phase, epoch_iter, batch_idx, timer)
                self.compute_metrics(phase, meters, outputs)

                postfix = {k: v.mean for k, v in meters.items()}

                for cb in self.batch_callbacks:
                    cb(phase, epoch, batch_idx, postfix)

                if progress:
                    epoch_progress.set_postfix(postfix)

                self.after_batch(phase, epoch, batch_idx)

        metrics = {k: v.mean for k, v in meters.items()}

        if phase == "train":
            metrics["lr"] = self.optim.param_groups[0]["lr"]
            if timing:
                for k, t in timer.measurements:
                    metrics[f"{k}_mean"] = t.mean
                    metrics[f"{k}_std"] = t.std

        metrics.update(dict(epoch=epoch, phase=phase))
        self.metrics.dump(metrics)
        self.after_epoch(phase, epoch)

        return metrics

    def run_step(self, phase, batch_iter, batch_idx, timer):
        with timer("t_data"):
            x, y = to_device(next(batch_iter), self.device)
        with timer("t_forward"):
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)
        if phase == "train":
            with timer("t_backward"):
                loss.backward()
            with timer("t_optim"):
                if isinstance(self.scheduler, WarmupScheduler):
                    self.scheduler.warmup_step()
                if (
                    self.accumulate_gradients is None
                    or (batch_idx + 1) % self.accumulate_gradients == 0
                ):
                    self.optim.step()
                    self.optim.zero_grad()

        return dict(loss=loss, y=y, yhat=yhat)

    def compute_metrics(self, phase, meters, outputs):
        loss, y, yhat = outputs["loss"], outputs["y"], outputs["yhat"]
        meters["loss"].add(loss.item())
        if self.metric_fns:
            for name, f in self.metric_fns.items():
                val = f(yhat, y)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                meters[name].add(val)

    def before_batch(self, phase, epoch, batch_idx):
        pass

    def after_batch(self, phase, epoch, batch_idx):
        pass

    def before_epoch(self, phase, epoch):
        pass

    def after_epoch(self, phase, epoch):
        pass

    def train(self, epoch=0):
        return self.run_epoch("train", epoch)

    def eval(self, epoch=0):
        return self.run_epoch("val", epoch)

    def test(self, epoch=0):
        return self.run_epoch("test", epoch)

    def run(self, resume=False):
        if not resume:
            self.to_device()
            self.build_logging()
            printc(f"Running {str(self)}", color="YELLOW")
            self.run_epochs()
        else:
            self.load()
            self.resume()

    def resume(self):
        last_epoch = self._epoch
        printc(f"Resuming from start of epoch {last_epoch}", color="YELLOW")
        printc(f"Running {str(self)}", color="YELLOW")
        self.run_epochs(start=last_epoch)
