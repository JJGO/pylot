from dataclasses import dataclass
from functools import wraps
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class SelfsuperDataset(Dataset):
    def __init__(self, inner_dataset: Dataset):
        self.inner_dataset = inner_dataset

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        x = self.inner_dataset[index]
        if isinstance(x, tuple):
            x = x[0]
        return x, x

    def __len__(self) -> int:
        return len(self.inner_dataset)  # type: ignore


def self_supervised(dataset: Dataset) -> Dataset:
    def duplicate(f):
        @wraps(f)
        def duplicated(index):
            x = f(index)
            if isinstance(x, tuple):
                x = x[0]
            return x, x

        return duplicated

    dataset.__getitem__ = duplicate(dataset.__getitem__)  # type: ignore
    return dataset
