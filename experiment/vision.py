import copy

from pylot.models.vision import replace_head, get_classifier_module

from .train import TrainExperiment
from ..metrics import correct


class VisionClassificationTrainExperiment(TrainExperiment):
    """Vision Classification TrainExperiment
    """

    def build_model(self, model, **model_kwargs):
        # Replace classifier layer (i.e. pre-softmax)
        # So #-of-classes is correct
        super().build_model(model, **model_kwargs)
        clf = get_classifier_module(self.model)
        if clf.out_features != self.dataset.n_classes:
            replace_head(self.model, self.dataset.n_classes)
        clf = get_classifier_module(self.model)
        assert clf.out_features == self.dataset.n_classes

    def compute_metrics(self, phase, meters, loss, y, yhat):
        c1, c5 = correct(yhat, y, (1, 5))
        batch_size = getattr(self, f"{phase}_dl").batch_size
        meters[f"{phase}_acc1"].add(c1 / batch_size)
        meters[f"{phase}_acc5"].add(c5 / batch_size)

    def build_data(self, **kwargs):
        # Ensure validation set is not augmented
        super().build_data(**kwargs)
        self.val_dataset = copy.deepcopy(self.val_dataset)
        self.val_dataset.transform = self.test_dataset.transform
