import collections
import pathlib
from contextlib import contextmanager
from typing import Iterable

import lmdb
from lmdbm import Lmdb

from pydantic import validate_arguments

from pylot.util import autopackb, autounpackb


class ThunderDB(Lmdb):
    def _pre_key(self, key: str) -> bytes:
        return key.encode()

    def _post_key(self, key: bytes) -> str:
        return key.decode()

    def _pre_value(self, value: object) -> bytes:
        return autopackb(value)

    def _post_value(self, value: bytes) -> object:
        return autounpackb(value)


class ThunderDict(collections.abc.MutableMapping):
    @validate_arguments
    def __init__(self, path: pathlib.Path):
        self.path = path

    @contextmanager
    def db(self):
        with ThunderDB.open(str(self.path), "c") as db:
            yield db

    def __getitem__(self, key):
        with self.db() as db:
            return db[key]

    def __setitem__(self, key, value):
        with self.db() as db:
            db[key] = value

    def __len__(self):
        with self.db() as db:
            return len(db)

    def __iter__(self):
        with self.db() as db:
            return iter([k for k in db.keys()])

    def __contains__(self, key):
        with self.db() as db:
            return key in db

    def __delitem__(self, key):
        with self.db() as db:
            del db[key]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(str(self.path))})"

    def update(self, other):
        with self.db() as db:
            for key, val in other.items():
                db[key] = val


class ThunderReader(collections.abc.Mapping):
    @validate_arguments
    def __init__(self, path: pathlib.Path):
        self.path = path
        self._env = None
        self._txn = None

    def _require_env(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._txn = self._env.begin(write=False)

    def __getitem__(self, key):
        self._require_env()
        return autounpackb(self._txn.get(key.encode()))

    def keys(self):
        self._require_env()
        for key, value in self._txn.cursor().iternext(keys=True, values=True):
            yield key.decode()

    def __len__(self):
        self._require_env()
        return self._txn.stat()["entries"]

    def __iter__(self):
        return self.keys()

    def __repr__(self):
        return f'{self.__class__.__name__}("{str(self.path)}")'


#     def pget(self, keys: Iterable[str], max_workers=8):
#         from concurrent.futures import ProcessPoolExecutor
#         global _thunder_load # needed to trick concurrent executor
#         def _thunder_load(key):
#             return key, autounpackb(self._txn.get(key.encode()))
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             data = dict(executor.map(_thunder_load, keys))
#         del _thunder_load
#         return data


