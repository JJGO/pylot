import pathlib
from typing import List

from pydantic import validate_arguments

from torch.utils.data import Dataset

from ..util import ThunderLoader, ThunderReader, UniqueThunderReader


class ThunderDataset(Dataset):
    @validate_arguments
    def __init__(
        self, path: pathlib.Path, preload: bool = False, reuse_fp: bool = True
    ):
        self._path = path
        self.preload = preload
        if preload:
            self._db = ThunderLoader(path)
        elif reuse_fp:
            self._db = UniqueThunderReader(path)
        else:
            self._db = ThunderReader(path)

        if "samples" in self._db.keys():
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

    @staticmethod
    def supress_readonly_warning():
        """
        Ignores warnings about non-writable numpy arrays
        when constructing torch objects. This is ok, 99%
        of the time since data is not backprop'ed or 
        modified in place (e.g. changing dtype forces copy).
        Still, not called by default, subclass should decide
        whether to call it upon __init__
        """
        import warnings

        msg = """The given NumPy array is not writable.*"""
        warnings.filterwarnings("ignore", msg, UserWarning)
