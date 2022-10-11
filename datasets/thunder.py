import pathlib
from typing import List

from torch.utils.data import Dataset
from pydantic import validate_arguments

from ..util import UniqueThunderReader, ThunderLoader


class ThunderDataset(Dataset):
    @validate_arguments
    def __init__(self, path: pathlib.Path, preload: bool = False):
        self._path = path
        self.preload = preload
        if preload:
            self._db = ThunderLoader(path)
        else:
            self._db = UniqueThunderReader(path)

        self.samples: List[str] = self._db["_samples"]
        self.attrs = self._db.get("_attrs", {})

    def _load(self, key):
        true_key = self.samples[key]
        data = self._db[true_key]
        return data

    def __getitem__(self, key):
        return self._load(key)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(str(self._path))})"
