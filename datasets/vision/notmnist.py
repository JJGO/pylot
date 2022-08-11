from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
from PIL import Image
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder, MNIST

from tqdm.auto import tqdm

from pylot.datasets import DatapathMixin, IndexedImageFolder

# Download and untar
# http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
# http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz

# NOTE:
# There are invalid images that need to be deleted, and the Index might need to be recreated several times


def _load_sample(path):
    #     try:
    return Image.open(path).convert("L")


#     except PIL.UnidentifiedImageError:
#         return path


class notMNIST(MNIST):

    classes = list("ABCDEFGHIJ")

    # loading code is a tad hacky to make the dataset MNIST-inherited

    @validate_arguments
    def __init__(
        self,
        root: Path,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
        preload: bool = True,
    ):
        if download:
            raise NotImplementedError
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.preload = preload
        self._dset = IndexedImageFolder(root / self._folder)
        idxs = np.arange(len(self._dset.imgs))
        train_idx, test_idx = train_test_split(
            idxs, random_state=42, test_size=0.2, stratify=self._dset.targets
        )
        self.split_idx = train_idx if train else test_idx

        # [...,0] is because images are loaded as RGB but are Greyscale
        # self.data = [torch.from_numpy(np.array(dset[i][0])[..., 0]) for i in split_idx]
        self.targets = torch.from_numpy(np.array(self._dset.targets))[self.split_idx]
        self.root = self._dset.root

        if self.preload:
            self._load_data()

    def _load_data(self):
        paths = [self._dset.imgs[i][0] for i in self.split_idx]
        with ProcessPoolExecutor(max_workers=16) as executor:
            self.data = list(
                tqdm(executor.map(_load_sample, paths), leave=False, total=len(paths))
            )

    def __getitem__(self, idx):
        if self.preload:
            x = self.data[idx]
            y = self.targets[idx].item()
        else:
            idx = self.split_idx[idx]
            x, y = self._dset[idx]
            x = x.convert("L")
        x = self.transform(x) if self.transform else x
        y = self.target_transform(y) if self.target_transform else y
        return x, y

    def __len__(self):
        return len(self.targets)

    @property
    def _folder(self):
        return "notMNIST/notMNIST_small"


class notMNISTLarge(notMNIST):
    @property
    def _folder(self):
        return "notMNIST/notMNIST_large"
