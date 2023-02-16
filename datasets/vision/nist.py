import functools
import pathlib
import os
from typing import Literal

from torchvision import transforms, datasets
import torchvision.transforms as TF
import torchvision.datasets as TD
import PIL
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
import numpy as np

from ..indexed import IndexedImageDataset
from ..path import DatapathMixin
from .emnist import EMNIST
from .usps28 import USPS28
from .notmnist import notMNIST, notMNISTLarge


__all__ = [
    "MNIST",
    "EMNIST",
    "KMNIST",
    "USPS28",
    "FashionMNIST",
    "notMNIST",
    "notMNISTLarge",
]

DATASET_CONSTRUCTOR = {
    "MNIST": TD.MNIST,
    "KMNIST": TD.KMNIST,
    "EMNIST": EMNIST,
    "FashionMNIST": TD.FashionMNIST,
    "USPS": TD.USPS,
    "USPS28": USPS28,
    "notMNIST": notMNIST,
    "notMNISTLarge": notMNISTLarge,
    "CIFAR10": TD.CIFAR10,
    "CIFAR100": TD.CIFAR100,
    "ImageNet": IndexedImageDataset,
    "Places365": IndexedImageDataset,
    "Miniplaces": IndexedImageDataset,
    "TinyImageNet": IndexedImageDataset,
}

DATASET_NCLASSES = {
    "MNIST": 10,
    "FashionMNIST": 10,
    "EMNIST": 26,
    "KMNIST": 10,
    "USPS": 10,
    "USPS28": 10,
    "CIFAR10": 10,
    "SVHN": 10,
    "STL10": 10,
    "LSUN": 10,
    "TinyImageNet": 200,
    "notMNIST": 10,
    "notMNISTLarge": 10,
}

DATASET_SIZES = {
    "MNIST": (28, 28),
    "FashionMNIST": (28, 28),
    "EMNIST": (28, 28),
    "QMNIST": (28, 28),
    "KMNIST": (28, 28),
    "notMNIST": (28, 28),
    "USPS": (16, 16),
    "USPS28": (28, 28),
    "SVHN": (32, 32),
    "CIFAR10": (32, 32),
    "STL10": (96, 96),
    "TinyImageNet": (64, 64),
}

DATASET_NORMALIZATION = {
    "MNIST": ((0.1307,), (0.3081,)),
    "USPS": ((0.2459,), (0.2977,)),
    "USPS28": ((0.2459,), (0.2977,)),
    "FashionMNIST": ((0.2849,), (0.3516,)),
    "QMNIST": ((0.1307,), (0.3081,)),
    "EMNIST": ((0.1716,), (0.3297,)),
    "KMNIST": ((0.1910,), (0.3470,)),
    "notMNIST": ((0.419,), (0.455,)),
    "notMNISTLarge": ((0.413,), (0.451,)),
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "TinyImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR10": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "STL10": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _fromVisionDataset(dataset_name):

    class_ = DATASET_CONSTRUCTOR[dataset_name]

    class DatasetWrapper(class_, DatapathMixin):
        @validate_arguments
        def __init__(
            self,
            split: Literal["train", "val", "test"],
            val_split: float = 0.0,
            seed: int = 42,
            download: bool = False,
            augmentation: bool = True,
            transforms: bool = True,
        ):
            self.split = split
            self.val_split = val_split
            self.seed = seed
            self.augmentation = augmentation
            if split == "val" and val_split == 0.0:
                raise ValueError("val_split cannot be 0.0 when split is 'val'")

            aug_transforms = [TF.RandomCrop(28, padding=4)] if self.augmentation else []
            transform = TF.Compose(aug_transforms + [
                TF.ToTensor(),
                TF.Normalize(*self.normalization)
            ]) if transforms else None
            root = self.path.parent if dataset_name not in ("USPS",) else self.path
            super().__init__(
                root, transform=transform, train=(split != "test"), download=download
            )
            self._init_split()

        def _init_split(self):
            if self.split == "test":
                self.indices = np.arange(len(self.targets))
            elif self.val_split == 0:
                if self.split == "train":
                    self.indices = np.arange(len(self.targets))
                else:
                    self.indices = []
            else:
                train_idx, val_idx = train_test_split(
                    np.arange(len(self.targets)),
                    random_state=self.seed,
                    test_size=self.val_split,
                    stratify=self.targets,
                )
                if self.split == "train":
                    self.indices = train_idx
                else:
                    self.indices = val_idx

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return super().__getitem__(self.indices[idx])

        @property
        def _folder_name(self):
            return dataset_name

        @property
        def normalization(self):
            return DATASET_NORMALIZATION[dataset_name]

        @property
        def size(self):
            return DATASET_SIZES[dataset_name]

        @property
        def n_classes(self):
            return DATASET_NCLASSES[dataset_name]

    return type(dataset_name, (DatasetWrapper,), {})


MNIST = _fromVisionDataset("MNIST")
EMNIST = _fromVisionDataset("EMNIST")
KMNIST = _fromVisionDataset("KMNIST")
USPS28 = _fromVisionDataset("USPS28")
FashionMNIST = _fromVisionDataset("FashionMNIST")
notMNIST = _fromVisionDataset("notMNIST")
notMNISTLarge = _fromVisionDataset("notMNISTLarge")
