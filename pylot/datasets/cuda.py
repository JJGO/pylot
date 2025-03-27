import torch
from torch.utils.data import Dataset

from ..util import to_device


class CUDACachedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        assert torch.cuda.is_available()
        self._dataset = dataset
        self._cache = [to_device(i, "cuda") for i in self._dataset]

    def __getitem__(self, idx):
        return self._cache[idx]

    def __getattr__(self, key):
        # This works because __getattr__ is only called as last resort
        # https://stackoverflow.com/questions/2405590/how-do-i-override-getattr-without-breaking-the-default-behavior
        return getattr(self._dataset, key)

    def __len__(self):
        return len(self._cache)
