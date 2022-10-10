import copy
import hashlib
import pathlib
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from typing import Any, Union, Tuple, Dict

import yaml

from .ioutil import autoload, autosave
from .more_functools import newobj

NormalizedKey = Tuple[Union[str, int], ...]
Key = Union[str, NormalizedKey]


def keymap(func, d):
    return {func(k): v for k, v in d.items()}


def valmap(func, d):
    return {k: func(v) for k, v in d.items()}


def parse_key(key, sep=None) -> NormalizedKey:
    if isinstance(key, str):
        if sep is not None:
            return tuple(key.split(sep))
        return (key,)
    elif isinstance(key, (tuple, list)):
        return sum((parse_key(subkey, sep=sep) for subkey in key), start=tuple())
    elif isinstance(key, int):
        return (key,)
    else:
        raise TypeError(f"Key must be str|int|Tuple[str], not {type(key)}")


def flatten(nested_dict, sep=None) -> Dict[Key, Any]:
    flat_dict = {}
    for k, v in nested_dict.items():
        if isinstance(v, Mapping):
            for k2, v2 in flatten(v).items():
                flat_dict[(k,) + k2] = v2
        else:
            flat_dict[(k,)] = v
    if sep is not None:
        flat_dict = keymap(lambda k: sep.join(k), flat_dict)
    return flat_dict


def deepupdate(original, update) -> Dict[Key, Any]:
    for k, v in update.items():
        if isinstance(v, Mapping):
            original[k] = deepupdate(original.get(k, {}), v)
        else:
            original[k] = v
    return original


def unflatten(flat_dict, sep=None):
    nested_dict = {}
    for k, v in flat_dict.items():
        set_nested(nested_dict, k, v, sep=sep)
    return nested_dict


def _valid_list_key(key, list_) -> bool:
    if not isinstance(key, (str, int)):
        return False
    if isinstance(key, str):
        if not key.isdecimal():
            return False
        key = int(key)
    return -len(list_) <= key < len(list_)


def get_nested(nested_dict, key, sep=None):
    key = parse_key(key, sep=sep)
    key_so_far = tuple()
    d = nested_dict
    for subkey in key:
        key_so_far += (subkey,)
        if isinstance(d, list):
            if not _valid_list_key(subkey, d):
                raise KeyError(key_so_far)
            subkey = int(subkey)
        if isinstance(d, dict) and subkey not in d:
            raise KeyError(key_so_far)
        d = d[subkey]
    return d


def set_nested(nested_dict, key, value, sep=None):
    key = parse_key(key, sep=sep)
    *prefix, key = key
    d = nested_dict
    for parent in prefix:
        prev = d
        if isinstance(d, dict):
            d = d.setdefault(parent, {})
        elif isinstance(d, list):
            d = d[parent]
        # if intermediate key has non-dict value, we overwrite
        else:
            d = prev[parent] = {}
    if isinstance(d, list):
        assert _valid_list_key(key, d)
        key = int(key)
    d[key] = value


def pop_nested(nested_dict, key, sep=None) -> Any:
    *prefix, key = parse_key(key, sep=sep)
    d = nested_dict
    key_so_far = tuple()
    for subkey in prefix:
        key_so_far += (subkey,)
        if isinstance(d, list):
            if not _valid_list_key(subkey, d):
                raise KeyError(key_so_far)
            subkey = int(subkey)
        if isinstance(d, dict) and subkey not in d:
            raise KeyError(key_so_far)
        d = d[subkey]
    if isinstance(d, list):
        key = int(key)
    return d.pop(key)


def del_nested(nested_dict, key, sep=None):
    pop_nested(nested_dict, key, sep=sep)


def contains_nested(nested_dict, key, sep=None) -> bool:
    key = parse_key(key, sep=sep)
    d = nested_dict
    for subkey in key:
        if not isinstance(d, (list, dict)):
            return False
        elif isinstance(d, dict) and subkey not in d:
            return False
        elif isinstance(d, list):
            if not _valid_list_key(subkey, d):
                return False
            subkey = int(subkey)
        d = d[subkey]
    return True


class HDict(MutableMapping):
    # Hierarchical Dictionary

    def __init__(self, initial=None, *, _sep="."):
        self._sep = _sep
        self._data = dict()
        if initial is not None:
            if isinstance(initial, HDict):
                initial = initial.to_dict()
            initial = unflatten(initial, sep=self._sep)
            self._data = dict(initial)

    def __repr__(self):
        s = f"{self.__class__.__name__}({repr(self._data)}"
        if self._sep != ".":
            s += f", _sep={repr(self._sep)}"
        s += ")"
        return s

    def __contains__(self, key: Key) -> bool:
        return contains_nested(self._data, key, sep=self._sep)

    def __getitem__(self, key: Key):
        val = get_nested(self._data, key, sep=self._sep)
        if isinstance(val, dict):
            val = self.__class__(val, _sep=self._sep)
        return val

    # Composite methods
    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]

    # Delegated methods
    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # Methods are inherited from Mapping mixin
    # def keys(self):
    #     return self._data.keys()
    # def items(self):
    #     return self._data.items()
    # def values(self):
    #     return self._data.values()

    # Mutable methods
    def __setitem__(self, key: Key, value):
        return set_nested(self._data, key, value, sep=self._sep)

    def __delitem__(self, key: Key):
        return del_nested(self._data, key, sep=self._sep)

    def pop(self, key: Key, *default):
        assert len(default) <= 1, "Expected at most one default value"
        if key not in self and len(default) > 0:
            return default[0]
        val = pop_nested(self._data, key, sep=self._sep)
        if isinstance(val, dict):
            val = HDict(val, _sep=self._sep)
        return val

    def update(self, other: Union["HDict", dict]):
        if isinstance(other, HDict):
            other = other._data
        return deepupdate(self._data, HDict.from_flat(other, sep=self._sep))

    def clear(self):
        return self._data.clear()

    # Derived methods
    def set(self, key: Key, value):
        self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    # Custom methods
    def flatten(self) -> dict:
        return flatten(self._data, sep=self._sep)

    @classmethod
    def from_flat(cls, flat_dict, sep=".") -> "HDict":
        return cls(unflatten(flat_dict, sep=sep))

    @classmethod
    def from_file(cls, path) -> "HDict":
        return cls(autoload(path))

    def as_yaml(self) -> str:
        return yaml.safe_dump(self._data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def copy(self) -> "HDict":
        return self.__class__(copy.deepcopy(self._data))


class ImmutableConfigError(Exception):
    def __init__(self):
        super().__init__("Config objects are inmutable")


def config_digest(config):
    return hashlib.md5(
        yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


class Config(HDict):
    def __hash__(self):
        return hash(int(self.digest(), 16))

    def digest(self):
        return config_digest(self._data)

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Item assignment must be done via .set because Config is a functional data structure"
        )

    def __delitem__(self, key, value):
        raise NotImplementedError(
            "Item removal must be done via .remove because Config is a functional data structure"
        )

    @newobj
    def set(self, key, value):
        super().__setitem__(key, value)

    @newobj
    def remove(self, key):
        super().__delitem__(key)

    @newobj
    def pop(self, key, *default):
        return super().pop(key, *default)

    @newobj
    def update(self, other):
        return super().update(other)


class ImmutableConfig(HDict):
    def __hash__(self):
        return hash(int(self.digest(), 16))

    def digest(self):
        return config_digest(self._data)

    def __setitem__(self, key, value):
        raise ImmutableConfigError()

    def __delitem__(self, key):
        raise ImmutableConfigError()

    def pop(self, key):
        raise ImmutableConfigError()

    def update(self, other):
        raise ImmutableConfigError()


# Mutablde File-Backed dictionary


class FHDict(HDict):
    # File-backed Hierarchical Dictionary

    def __init__(self, path: Union[str, pathlib.Path], _sep=".", prefix=tuple()):
        self._path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        assert self.path.suffix in (
            ".json",
            ".yaml",
            ".yml",
        ), f"Invalid extension {self.path.suffix}, file must be JSON or YAML"
        self._sep = _sep
        self._prefix = prefix
        if not self._path.exists():
            autosave({}, self.path)

    @property
    def root(self):
        return self.__class__(self._path, _sep=self._sep)

    @property
    def path(self):
        return self._path

    @property
    def prefix(self):
        return self._prefix

    @contextmanager
    def _fileop(self, persist=False):
        data = autoload(self._path)
        self._data = get_nested(data, self._prefix, sep=self._sep)
        yield
        if persist:
            autosave(data, self._path)
        delattr(self, "_data")

    def __contains__(self, key: Key) -> bool:
        with self._fileop():
            return super().__contains__(key)

    def __getitem__(self, key: Key):
        with self._fileop():
            val = get_nested(self._data, key, sep=self._sep)
            if isinstance(val, dict):
                val = self.__class__(self._path, prefix=key, _sep=self._sep)
            return val

    def __iter__(self):
        with self._fileop():
            return super().__iter__()

    def __len__(self):
        with self._fileop():
            return super().__len__()

    # Mutable methods
    def __setitem__(self, key: Key, value):
        with self._fileop(persist=True):
            super().__setitem__(key, value)

    def __delitem__(self, key: Key):
        with self._fileop(persist=True):
            super().__delitem__(key)

    def pop(self, key: Key, *default):
        with self._fileop(persist=True):
            return super().pop(key)

    def clear(self):
        with self._fileop(persist=True):
            super().clear()

    def update(self, other: Union["HDict", dict]):
        with self._fileop(persist=True):
            super().update(other)

    def __repr__(self):
        s = f"{self.__class__.__name__}({repr(self._path)}"
        if self._prefix is not None:
            s += f", prefix={repr(self._prefix)}"
        if self._sep != ".":
            s += f", _sep={repr(self._sep)}"
        s += ")"
        return s

    def __str__(self):
        with self._fileop():
            header = f"# {repr(self)}\n---\n"
            return header + yaml.safe_dump(self._data)

    def to_dict(self):
        with self._fileop():
            return super().to_dict()


def check_missing(config):
    flat_cfg = Config(config).flatten()
    for key, value in flat_cfg.items():
        if value == "?":
            raise ValueError(f"Must provide a value for {key}")


def configdiff(*cfgs):
    from functools import reduce
    import operator
    import pandas as pd

    ds = [flatten(cfg) for cfg in cfgs]
    rows = []
    ks = reduce(operator.or_, [set(d.keys()) for d in ds])
    for k in ks:
        vs = [d.get(k, "") for d in ds]
        if not all(vs[0] == v for v in vs):
            rows.append([k] + vs)
    return pd.DataFrame(data=rows, columns=["key"] + [str(i) for i, _ in enumerate(ds)])
