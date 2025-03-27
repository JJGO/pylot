from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from ..path import DatapathMixin


class OxfordPets(Dataset, DatapathMixin):
    """
    https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    # TODO add keypoint support for bounding boxes
    MODES = ("cls2", "cls37", "seg1", "seg2", "seg37")
    MEAN_CH = (0.478, 0.443, 0.394)
    STD_CH = (0.268, 0.263, 0.27)

    def __init__(
        self,
        split,
        download=False,
        root=None,
        cache=False,
        preproc=False,
        mode="cls2",
        size: Tuple[int, int] = (192, 256),
        augmentation=False,
        onehot: bool = False,
    ):
        train = (split == 'train')
        assert mode in self.MODES, f"Mode {mode} not one of {','.join(self.MODES)}"
        assert not download or (
            root is not None
        ), "For download, root must be specified"

        self.mode = mode
        self.train = train
        self.breeds = "37" in self.mode
        self.root = root
        self.cache = cache
        self.preproc = preproc
        self.size = size
        self.augmentation = augmentation
        self.onehot = onehot
        self.n_classes = len(self.classes) + 1  # Extra channel for background

        if download:
            self._download()

        self._load_annotations()

        if self.preproc:
            self._load_transforms()

        if self.cache:
            if self.augmentation:
                self._get_image = lru_cache(maxsize=None)(self._get_image)
                self._get_mask = lru_cache(maxsize=None)(self._get_mask)
            else:
                self.__getitem__ = lru_cache(maxsize=None)(self.__getitem__)

    def _load_annotations(self):
        self.images_path = self.path / "images"
        self.masks_path = self.path / "annotations/trimaps"

        label_file = "trainval.txt" if self.train else "test.txt"
        label_df = pd.read_csv(
            self.path / "annotations" / label_file,
            sep=" ",
            names=["file", "id", "species", "breed"],
        )

        self._files = label_df.file.values
        labels = label_df.id if self.breeds else label_df.species
        # Labels are one indexed
        self._labels = labels.values - 1

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
        return len(self._files)

    def __getitem__(self, index):
        image = self._get_image(index)
        segment = self.mode.startswith("seg")
        if not segment:
            label = self._labels[index]
            if self.preproc:
                image = self.transform(image=image)["image"]
        else:
            label = self._get_mask(index)
            if self.preproc:
                augment = self.transform(image=image, mask=label)
                image, label = augment["image"], augment["mask"].long()
                if self.mode == "seg1":
                    # Dummy dimension for BCE(WithLogits)Loss or volumetrics losses
                    label = label[None, ...].float()
                if self.onehot:
                    label = (
                        F.one_hot(label, num_classes=self.n_classes)
                        .permute((2, 0, 1))
                        .float()
                    )

        return image, label

    def _get_image(self, i):
        # Convert is required in case RGBA
        image = Image.open(self.images_path / (self._files[i] + ".jpg")).convert("RGB")
        if self.preproc:
            image = np.array(image)
        return image

    def _get_mask(self, i):
        trimap = Image.open(self.masks_path / (self._files[i] + ".png"))
        mask = np.array(trimap)
        # Pixel Annotations: 1: Foreground 2:Background 3: Not classified
        mask[mask == 2] = 0.0
        mask[(mask == 1) | (mask == 3)] = 1.0

        if self.mode == "seg1":
            return mask

        mask *= 1 + self._labels[i]
        return mask

    def _download(self):
        base_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
        download_and_extract_archive(base_url + "images.tar.gz", self.root)
        download_and_extract_archive(base_url + "annotations.tar.gz", self.root)

    @property
    def classes(self):
        if self.mode == "seg1":
            return ["mask"]
        if not self.breeds:
            return ["cat", "dog"]
        else:
            return [
                "Abyssinian",
                "american_bulldog",
                "american_pit_bull_terrier",
                "basset_hound",
                "beagle",
                "Bengal",
                "Birman",
                "Bombay",
                "boxer",
                "British_Shorthair",
                "chihuahua",
                "Egyptian_Mau",
                "english_cocker_spaniel",
                "english_setter",
                "german_shorthaired",
                "great_pyrenees",
                "havanese",
                "japanese_chin",
                "keeshond",
                "leonberger",
                "Maine_Coon",
                "miniature_pinscher",
                "newfoundland",
                "Persian",
                "pomeranian",
                "pug",
                "Ragdoll",
                "Russian_Blue",
                "saint_bernard",
                "samoyed",
                "scottish_terrier",
                "shiba_inu",
                "Siamese",
                "Sphynx",
                "staffordshire_bull_terrier",
                "wheaten_terrier",
                "yorkshire_terrier",
            ]
