from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..path import DatapathMixin


class PascalVOC2012(Dataset, DatapathMixin):
    """
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    """

    MODES = ("seg20",)  # , "detect")
    MEAN_CH = (0.457, 0.443, 0.407)
    STD_CH = (0.273, 0.269, 0.285)

    def __init__(
        self,
        train: bool = True,
        preproc: bool = False,
        augmentation: bool = False,
        cache: bool = False,
        size: Tuple[int, int] = (192, 256),
        mode: str = "seg20",
        onehot: bool = False,
    ):
        assert mode in self.MODES, f"Mode {mode} not one of {','.join(self.MODES)}"

        self.cache = cache
        self.train = train
        self.preproc = preproc
        self.augmentation = augmentation
        self.size = size
        self.mode = mode
        self.onehot = onehot
        self.n_classes = len(self.classes)

        image_set = "train" if self.train else "val"
        self.D = VOCSegmentation(root=self.path, year="2012", image_set=image_set)

        if self.preproc:
            self._load_transforms()

        if self.cache:
            if self.augmentation:
                self.D.__getitem__ = lru_cache(maxsize=None)(self.D.__getitem__)
            else:
                self.__getitem__ = lru_cache(maxsize=None)(self.__getitem__)

    def _load_transforms(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        transforms = []
        if self.size:
            height, width = self.size

            if self.train and self.augmentation:
                transforms = [
                    A.RandomResizedCrop(height, width),
                    A.HorizontalFlip(p=0.5),
                ]
            else:
                transforms = [A.Resize(height, width)]

        # For either train/val we need to Normalize channels and convert to tensor
        transforms += [
            A.Normalize(mean=self.MEAN_CH, std=self.STD_CH),
            ToTensorV2(),
        ]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, index):
        image, label = self.D[index]
        if self.preproc:
            image = np.array(image)
            label = np.array(label)
            label[label == 255] = 0
            augment = self.transform(image=image, mask=label)
            image, label = augment["image"], augment["mask"].long()
            if self.onehot:
                label = (
                    F.one_hot(label, num_classes=self.n_classes)
                    .permute((2, 0, 1))
                    .float()
                )
        return image, label

    @property
    def classes(self):
        return [
            "__background__",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    @property
    def classes_dict(self):
        d = {i: l for i, l in enumerate(self.labels)}
        d[255] = "__void__"
        return d

    @property
    def palette(self):
        VOC_COLORMAP = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
        return np.array(VOC_COLORMAP)
