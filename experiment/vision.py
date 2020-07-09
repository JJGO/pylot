from collections import defaultdict

import torch
from tqdm import tqdm

from .train import TrainExperiment
from ..util import StatsMeter, CUDATimer
from ..metrics import correct
from pylot.models.vision import replace_head, get_classifier_module


class VisionClassificationTrainExperiment(TrainExperiment):
    """Vision Classification TrainExperiment
    """

    def build_model(self, model, **model_kwargs):
        super().build_model(model, **model_kwargs)
        clf = get_classifier_module(self.model)
        if clf.out_features != self.train_dataset.n_classes:
            replace_head(self.model, self.train_dataset.n_classes)
        clf = get_classifier_module(self.model)
        assert clf.out_features == self.train_dataset.n_classes

    def run_epoch(self, train, epoch=0):
        progress = self.get_param("log.progress", False)
        if train:
            self.model.train()
            phase = "train"
            dl = self.train_dl
        else:
            self.model.eval()
            phase = "val"
            dl = self.val_dl

        meters = defaultdict(StatsMeter)
        timer = CUDATimer(unit="ms", skip=10)
        if not self.get_param("log.timing", False):
            timer.disable()

        epoch_iter = iter(dl)
        if progress:
            epoch_progress = tqdm(epoch_iter)
            epoch_progress.set_description(
                f"{phase.capitalize()} Epoch {epoch}/{self.epochs}"
            )
            epoch_iter = iter(epoch_progress)

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
                    with timer("t_optim"):
                        self.optim.step()
                        self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                meters[f"{phase}_loss"].add(loss.item())
                meters[f"{phase}_acc1"].add(c1 / dl.batch_size)
                meters[f"{phase}_acc5"].add(c5 / dl.batch_size)

                postfix = {k: v.mean for k, v in meters.items()}

                for cb in self.batch_callbacks:
                    cb(self, postfix)

                if progress:
                    epoch_progress.set_postfix(postfix)

        if train and self.get_param("log.timing", False):
            self.log(timer.measurements)

        self.log(meters)
