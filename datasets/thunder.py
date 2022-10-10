import pathlib

from torch.utils.data import Dataset
from pydantic import validate_arguments

from ..util import ThunderReader


class ThunderDataset(Dataset):
    @validate_arguments
    def __init__(self, path: pathlib.Path, preload: bool = False):
        self._path = path
        self.preload = preload
        self._reader = None
        r = ThunderReader(self._path)
        self.samples = r["_samples"]
        self.attrs = r.get('_attrs', {})

        if self.preload:
            self._data = [self._load(i) for i in range(len(self))]
            delattr(self, "_reader")

    def _load(self, key):
        if self._reader is None:
            self._reader = ThunderReader(self._path)
        true_key = self.samples[key]
        data = self._reader[true_key]
        return data

    def __getitem__(self, key):
        if self.preload:
            return self._data[key]
        return self._load(key)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(str(self._path))})"
