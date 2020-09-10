from collections import defaultdict
import json
import pathlib

from tqdm.autonotebook import tqdm
import yaml

import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.optim
from torch.optim import lr_scheduler

import torchvision.datasets
import torchvision.models

from torchviz import make_dot

from .base import Experiment
from .util import any_getattr
from ..datasets import stratified_train_val_split
from ..log import summary
from ..loss import flatten_loss
from ..util import printc, StatsMeter, CUDATimer, allbut, get_full_env_info
from ..scheduler import WarmupScheduler
from .. import callbacks
from .. import datasets
from .. import models
from .. import loss as custom_loss


class TrainExperiment(Experiment):

    MODELS = [torchvision.models, models]
    DATASETS = [torchvision.datasets, datasets]
    CALLBACKS = [callbacks]
    LOSS = [torch.nn, custom_loss]
    OPTIMS = [torch.optim]
    SCHEDULERS = [lr_scheduler]

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
        self.batch_callbacks = None
        self.epoch_callbacks = None
        self.device = None

        self.build_data(**self.cfg["data"])
        self.build_model(**self.cfg["model"])
        self.build_loss(**self.cfg["loss"])
        self.build_train(**self.cfg["train"])

    def build_data(self, dataset, val_split=None, **data_kwargs):

        if hasattr(datasets, dataset):
            constructor = any_getattr(self.DATASETS, dataset)
            kwargs = allbut(data_kwargs, ["dataloader"])
            self.dataset = constructor(train=True, **kwargs)
            seed = self.get_param("experiment.seed")
            self.test_dataset = constructor(train=False, **kwargs)
            if val_split is not None:
                self.train_dataset, self.val_dataset = stratified_train_val_split(
                    self.dataset, val_split, seed=seed
                )
            else:
                self.val_dataset = self.test_dataset

        else:
            raise ValueError(f"Dataset {dataset} is not recognized")

        self.build_dataloader(**data_kwargs["dataloader"])

    def build_dataloader(self, **dataloader_kwargs):

        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, **dataloader_kwargs
        )
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)
        self.test_dl = DataLoader(self.test_dataset, shuffle=False, **dataloader_kwargs)

    def build_model(self, model, weights=None, **model_kwargs):
        constructor = any_getattr(self.MODELS, model)
        self.model = constructor(**model_kwargs)

        if weights is not None:
            self.load_model(weights)

    def build_loss(self, loss, flatten=False, **loss_kwargs):
        loss_func = any_getattr(self.LOSS, loss)(**loss_kwargs)
        if flatten:
            loss_func = flatten_loss(loss_func)

        self.loss_func = loss_func

    def build_train(self, optim, epochs, scheduler=None, warmup=None):

        self.epochs = epochs

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

    def _load_module(self, checkpoint, module):
        if isinstance(checkpoint, (str, pathlib.Path)):
            checkpoint = pathlib.Path(checkpoint)
            assert checkpoint.exists(), f"Checkpoint path {checkpoint} does not exist"
            checkpoint = torch.load(checkpoint)
        getattr(self, module).load_state_dict(checkpoint[f"{module}_state_dict"])

    def load_model(self, checkpoint):
        self._load_module(checkpoint, "model")

    def load_optim(self, checkpoint):
        self._load_module(checkpoint, "optim")

    def load_scheduler(self, checkpoint):
        self._load_module(checkpoint, "scheduler")

    def to_device(self):
        # Torch CUDA config
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

        torch.save(state, self.checkpoint_path / checkpoint_file)

    def load(self, tag=None):
        self.build_logging()
        self.to_device()
        # Load model & optimizer
        if not self.checkpoint_path.exists():
            printc("No checkpoints were found", color="ORANGE")
            self._epoch = 0
            return
        self.reload(tag=tag)

    def reload(self, tag=None):
        tag = tag if tag is not None else "last"
        checkpoint_file = f"{tag}.pt"
        checkpoint = torch.load(self.checkpoint_path / checkpoint_file)
        self._epoch = checkpoint["epoch"]
        self.load_model(checkpoint)
        self.load_optim(checkpoint)
        if self.scheduler is not None:
            self.load_scheduler(checkpoint)
        printc(
            f"Loaded checkpoint with tag:{tag}. Last epoch:{self._epoch}", color="BLUE"
        )

    def build_logging(self):
        super().build_logging()

        # Sample a batch
        x, y = next(iter(self.train_dl))
        # x, y = x.to(self.device), y.to(self.device)

        envinfo_path = self.path / "env.yml"
        with open(envinfo_path, "a+") as f:
            yaml.dump([get_full_env_info()], f)

        # Save model summary
        summary_path = self.path / "summary.txt"
        if self.get_param("log.summary", True) and not summary_path.exists():
            with open(summary_path, "w") as f:
                s = summary(self.model, x.shape[1:], echo=False, device="cpu")
                print(s, file=f)

                print("\n\nOptim\n", file=f)
                print(self.optim, file=f)

                if self.scheduler is not None:
                    print("\n\nScheduler\n", file=f)
                    print(self.scheduler, file=f)

            with open(summary_path.with_suffix(".json"), "w") as f:
                s = summary(
                    self.model, x.shape[1:], echo=False, device="cpu", as_stats=True
                )
                json.dump(s, f)

        # Save model topology
        topology_path = self.path / "topology"
        topology_pdf_path = topology_path.with_suffix(".pdf")
        if self.get_param("log.topology", False) and not topology_pdf_path.exists():
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)
            g = make_dot(loss)
            # g.format = 'svg'
            g.render(topology_path)
            # Interested in pdf, the graphviz file can be removed
            topology_path.unlink()

        del x
        del y

        # Callbacks
        self.batch_callbacks = []
        self.epoch_callbacks = []
        if "log" in self.cfg:
            if "batch_callbacks" in self.cfg["log"]:
                cbs = self.cfg["log"]["batch_callbacks"]
                self.batch_callbacks = [
                    any_getattr(self.CALLBACKS, k)(self, **args)
                    for c in cbs
                    for k, args in c.items()
                ]
            if "epoch_callbacks" in self.cfg["log"]:
                cbs = self.cfg["log"]["epoch_callbacks"]
                self.epoch_callbacks = [
                    any_getattr(self.CALLBACKS, k)(self, **args)
                    for c in cbs
                    for k, args in c.items()
                ]

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        try:
            for epoch in range(start, end):
                printc(f"Start epoch {epoch}", color="YELLOW")
                self._epoch = epoch
                self.checkpoint(tag="last")
                self.log(epoch=epoch)
                self.train(epoch)
                self.eval(epoch)
                if self.scheduler:
                    self.scheduler.step()

                with torch.no_grad():
                    for cb in self.epoch_callbacks:
                        cb(epoch)

                self.dump_logs()
            self.test(epoch)
            self.dump_logs()

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color="RED")
            self.checkpoint(tag="interrupt")
        self.checkpoint(tag="last")

    def run_epoch(self, train, epoch=0, test=False):
        progress = self.get_param("log.progress", False)
        timing = self.get_param("log.timing", False)
        if train:
            self.model.train()
            phase = "train"
            dl = self.train_dl
        else:
            self.model.eval()
            phase = "val"
            dl = self.val_dl
        if test:
            assert not train, "Cannot train and test at the same time"
            phase = "test"
            dl = self.test_dl

        meters = defaultdict(StatsMeter)
        timer = CUDATimer(unit="ms", skip=10)
        if not timing:
            timer.disable()

        epoch_iter = iter(dl)
        if progress:
            epoch_progress = tqdm(epoch_iter)
            epoch_progress.set_description(
                f"{phase.capitalize()} Epoch {epoch}/{self.epochs}"
            )
            epoch_iter = iter(epoch_progress)

        with torch.set_grad_enabled(train):
            for i in range(len(dl)):
                with timer("t_data"):
                    x, y = next(epoch_iter)
                    x, y = x.to(self.device), y.to(self.device)
                with timer("t_forward"):
                    yhat = self.model(x)
                    loss = self.loss_func(yhat, y)
                if train:
                    with timer("t_backward"):
                        loss.backward()
                    with timer("t_optim"):
                        if isinstance(self.scheduler, WarmupScheduler):
                            self.scheduler.warmup_step()
                        self.optim.step()
                        self.optim.zero_grad()

                meters[f"{phase}_loss"].add(loss.item())
                self.compute_metrics(phase, meters, loss, y, yhat)

                postfix = {k: v.mean for k, v in meters.items()}

                for cb in self.batch_callbacks:
                    cb(train, epoch, i, postfix)

                if progress:
                    epoch_progress.set_postfix(postfix)

        if train:
            self.log(lr=self.optim.param_groups[0]["lr"])
            if timing:
                self.log(timer.measurements)

        self.log(meters)

    def compute_metrics(self, phase, meters, loss, y, yhat):
        pass

    def train(self, epoch=0):
        return self.run_epoch(True, epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch)

    def test(self, epoch=0):
        return self.run_epoch(False, epoch, test=True)

    def run(self):
        self.build_logging()
        self.to_device()
        printc(f"Running {str(self)}", color="YELLOW")
        self.run_epochs()

    def resume(self):
        last_epoch = self._epoch
        printc(f"Resuming from start of epoch {last_epoch}", color="YELLOW")
        printc(f"Running {str(self)}", color="YELLOW")
        self.run_epochs(start=last_epoch)
