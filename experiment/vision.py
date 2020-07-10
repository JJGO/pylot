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

    def compute_metrics(self, phase, meters, loss, y, yhat):
        c1, c5 = correct(yhat, y, (1, 5))
        batch_size = getattr(self, f"{phase}_dl").batch_size
        meters[f"{phase}_acc1"].add(c1 / batch_size)
        meters[f"{phase}_acc5"].add(c5 / batch_size)
