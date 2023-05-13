from functools import lru_cache
from typing import Tuple
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

from ..path import DatapathMixin


class CUBBirds(Dataset, DatapathMixin):
    """
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    """

    MODES = ("cls", "seg")
    MEAN_CH = (0.486, 0.5, 0.433)
    STD_CH = (0.232, 0.228, 0.267)

    def __init__(
        self,
        split,
        download=False,
        root=None,
        cache=False,
        preproc=False,
        mode="cls",
        size: Tuple[int, int] = (192, 256),
        augmentation=False,
    ):
        train = (split == 'train')

        assert mode in self.MODES, f"Mode {mode} not one of {','.join(self.MODES)}"
        assert not download or (
            root is not None
        ), "For download, root must be specified"

        self.mode = mode
        self.train = train
        self.root = root
        self.cache = cache
        self.preproc = preproc
        self.size = size
        self.augmentation = augmentation

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
        def load_txt(file):
            return pd.read_csv(
                self.path / "CUB_200_2011" / file, index_col=0, sep=" ", names=["data"]
            )

        selector = load_txt("train_test_split.txt").data == int(self.train)
        self._files = load_txt("images.txt")[selector].data.values
        self._labels = load_txt("image_class_labels.txt")[selector].data.values - 1
        self._classes = load_txt("classes.txt").data.tolist()
        self.images_path = self.path / "CUB_200_2011/images"

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

    @property
    def classes(self):
        return self._classes

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        image = self._get_image(index)
        if self.mode == "cls":
            label = self._labels[index]
            if self.preproc:
                image = self.transform(image=image)["image"]
        else:
            label = self._get_mask(index)
            if self.preproc:
                augment = self.transform(image=image, mask=label)
                image, label = augment["image"], augment["mask"]
                label = label[None, ...].float()

        return image, label

    def _get_image(self, i):
        image = Image.open(self.path / "CUB_200_2011/images" / self._files[i]).convert(
            "RGB"
        )
        if self.preproc:
            image = np.array(image)
        return image

    def _get_mask(self, i):
        # Mask images have 5 grey levels (I believe this corresponds to 5 workers)
        # We consider 1 when at least 3 workers agree, 0 otherwise

        seg = Image.open(
            (self.path / "segmentations" / self._files[i]).with_suffix(".png")
        ).convert("L")
        # return seg
        mask = np.array(seg, dtype=np.uint8) // 51
        mask[mask < 3] = 0
        mask[mask >= 3] = 1
        if self.preproc:
            return mask
        return Image.fromarray(mask * 255)

    def _download(self):
        download_file_from_google_drive(
            "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45",
            self.root,
            filename="CUB_200_2011.tgz",
        )
        download_file_from_google_drive(
            "1EamOKGLoTuZdtcVYbHMWNpkn3iAVj8TP",
            self.root,
            filename="segmentations.tgz",
        )
        extract_archive(os.path.join(self.root, "CUB_200_2011.tgz"))
        extract_archive(os.path.join(self.root, "segmentations.tgz"))
