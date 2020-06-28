import pathlib
import json

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn

# from torchvision import datasets as tv_datasets
from torchviz import make_dot

from .base import Experiment
from .util import allbut, any_getattr
from ..log import summary
from ..loss import flatten_loss
from ..util import printc, StatsMeter, StatsTimer
from .. import callbacks
from .. import datasets
from .. import models
from .. import loss


class TrainExperiment(Experiment):

    MODELS = [models]
    DATASETS = [datasets]
    CALLBACKS = [callbacks]
    LOSS = [loss]

    def __init__(self, cfg=None, **kwargs):

        # Default children kwargs
        super().__init__(cfg, **kwargs)

        self.build_data(**self.cfg['data'])
        self.build_model(**self.cfg['model'])
        self.build_loss(**self.cfg['loss'])
        self.build_train(**self.cfg['train'])

    def build_data(self, dataset, **data_kwargs):

        if hasattr(datasets, dataset):
            constructor = any_getattr(self.DATASETS, dataset)
            kwargs = allbut(data_kwargs, ['dataloader'])
            self.train_dataset = constructor(train=True, **kwargs)
            self.val_dataset = constructor(train=False, **kwargs)

        else:
            raise ValueError(f"Dataset {dataset} is not recognized")

        self.build_dataloader(self, **data_kwargs['dataloader'])

    def build_dataloader(self, **dataloader_kwargs):

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dataloader_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)

    def build_model(self, model, weights=None, **model_kwargs):
        if hasattr(models, model):
            constructor = any_getattr(self.MODELS, model)
            self.model = constructor(**model_kwargs)
        else:
            raise ValueError(f"Architecture {model} not found")

        if weights is not None:
            self.load_model(weights)

    def build_loss(self, loss, flatten=False, **loss_kwargs):
        if hasattr(nn, loss):
            loss_func = any_getattr(self.LOSS, loss)(**loss_kwargs)
        if flatten:
            loss_func = flatten_loss(loss_func)

        self.loss_func = loss_func

    def build_train(self, optim, epochs, optim_state=None, **optim_kwargs):

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim

        if optim_state is not None:
            self.load_optim(optim_state)

    def load_model(self, checkpoint):
        if isinstance(checkpoint, (str, pathlib.Path)):
            checkpoint = pathlib.Path(checkpoint)
            assert checkpoint.exists(), f"Checkpoint path {checkpoint} does not exist"
            checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def load_optim(self, checkpoint):
        if isinstance(checkpoint, (str, pathlib.Path)):
            checkpoint = pathlib.Path(checkpoint)
            assert checkpoint.exists(), f"Checkpoint path {checkpoint} does not exist"
            checkpoint = torch.load(checkpoint)
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

    def checkpoint(self, epoch=None):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n if epoch is None else epoch
        checkpoint_file = f'checkpoint-{epoch:03d}.pt'
        if not (checkpoint_path / checkpoint_file).exists():
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict()
            }, checkpoint_path / checkpoint_file)
            with open(checkpoint_path / 'state.json', 'w') as f:
                json.dump({'epoch': epoch, 'file': checkpoint_file}, f)

    def load(self):
        self.to_device()
        self.build_logging()
        # Load model & optimizer
        checkpoint_path = self.path / 'checkpoints'
        if not checkpoint_path.exists():
            self.log_epoch_n = 0
            printc("No checkpoints were found", color='ORANGE')
            return

        with open(checkpoint_path / 'state.json', 'r') as f:
            state = json.load(f)

        self.log_epoch_n = state['epoch']
        checkpoint = torch.load(checkpoint_path / state['file'])
        self.load_model(checkpoint)
        self.load_optim(checkpoint)

    def build_logging(self):
        super().build_logging()

        # Sample a batch
        x, y = next(iter(self.train_dl))
        x, y = x.to(self.device), y.to(self.device)

        # Save model summary
        summary_path = self.path / 'summary.txt'
        if not summary_path.exists():
            with open(summary_path, 'w') as f:
                s = summary(self.model, x.shape[1:], echo=False)
                print(s, file=f)

                print("\n\nOptim\n", file=f)
                print(self.optim, file=f)

        # Save model topology
        topology_path = self.path / 'topology'
        if not topology_path.with_suffix('.pdf').exists():
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)
            g = make_dot(loss)
            # g.format = 'svg'
            g.render(topology_path)
            # Interested in pdf, the graphviz file can be removed
            topology_path.unlink()

        # Callbacks
        self.batch_callbacks = []
        self.epoch_callbacks = []
        if 'log' in self.cfg:
            if 'batch_callbacks' in self.cfg['log']:
                cbs = self.cfg['log']['batch_callbacks']
                self.batch_callbacks = [any_getattr(self.CALLBACKS, k)(self, **args) for c in cbs for k, args in c.items()]
            if 'epoch_callbacks' in self.cfg['log']:
                cbs = self.cfg['log']['epoch_callbacks']
                self.epoch_callbacks = [any_getattr(self.CALLBACKS, k)(self, **args) for c in cbs for k, args in c.items()]

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        try:
            for epoch in range(start, end):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.eval(epoch)

                with torch.set_grad_enabled(False):
                    for cb in self.epoch_callbacks:
                        cb(self, epoch)

                self.log_epoch(epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')
            self.checkpoint(self.log_epoch_n-1)

    def run_epoch(self, train, epoch=0):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            self.model.eval()
            prefix = 'val'
            dl = self.val_dl

        total_loss = StatsMeter()

        epoch_progress = tqdm(dl)
        epoch_progress.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")
        epoch_iter = iter(epoch_progress)

        timer = StatsTimer()

        with torch.set_grad_enabled(train):
            for _ in range(len(dl)):
                with timer("t_data"):
                    x, y = next(epoch_iter)
                    x, y = x.to(self.device), y.to(self.device)
                with timer("t_forward"):
                    yhat = self.model(x)
                    loss = self.loss_func(yhat, y)
                if train:
                    with timer("t_backward"):
                        loss.backward()

                        self.optim.step()
                        self.optim.zero_grad()

                total_loss.add(loss.item() / dl.batch_size)
                postfix = {'loss': total_loss.mean}

                for cb in self.batch_callbacks:
                    cb(self, postfix)
                epoch_progress.set_postfix(postfix)

        self.log({
            f'{prefix}_loss': total_loss.mean,
        }, timer.measurements)

        return total_loss.mean

    def train(self, epoch=0):
        return self.run_epoch(True, epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch)

    def run(self):
        self.to_device()
        self.build_logging()
        printc(f"Running {str(self)}", color='YELLOW')
        self.run_epochs()

    def resume(self):
        last_epoch = self.log_epoch_n
        printc(f"Resuming from epoch {last_epoch}\n {str(self)}", color='YELLOW')
        self.run_epochs(start=last_epoch+1)
