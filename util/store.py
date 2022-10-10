import collections
import pathlib
from contextlib import contextmanager

from lmdbm import Lmdb
from .ioutil import autoencode, autodecode


class AutoStore(collections.abc.MutableMapping):
    def __init__(self, path):
        self._path = pathlib.Path(path)

    @contextmanager
    def db(self):
        with Lmdb.open(str(self._path), "c") as db:
            yield db

    @staticmethod
    def _normalize_key(key):
        if isinstance(key, tuple):
            assert all("/" not in k for k in key)
            key = "/".join(key)
        key = pathlib.Path(key)
        if key.suffix == "":
            key = key.with_suffix(".pkl.lz4")
        return str(key)

    def __getitem__(self, key):
        key = self._normalize_key(key)
        with self.db() as db:
            return autodecode(db[key.encode("utf-8")], key)

    def __setitem__(self, key, value):
        key = self._normalize_key(key)
        with self.db() as db:
            db[key.encode("utf-8")] = autoencode(value, key)

    def __len__(self):
        with self.db() as db:
            return len(db)

    def __iter__(self):
        with self.db() as db:
            return iter([k.decode("utf-8") for k in db.keys()])

    def __contains__(self, key):
        key = self._normalize_key(key)
        with self.db() as db:
            return key in db

    def __delitem__(self, key):
        key = self._normalize_key(key)
        with self.db() as db:
            del db[str(key).encode("utf-8")]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(str(self._path))})"

    def update(self, other):
        with self.db() as db:
            for k, v in other.items():
                k = self._normalize_key(k)
                db[k.encode("utf-8")] = autoencode(v, k)
