import yaml
import hashlib
from collections.abc import Mapping, MutableMapping

# from .util import dict_recursive_update, expand_dots, expand_keys
from ..util import allbut, expand_keys


class Config(MutableMapping):
    def __init__(self, **kwargs):
        self.cfg = kwargs

    def __getitem__(self, key):
        if "." in key:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cfg})"

    def __str__(self):
        return yaml.safe_dump(self.cfg, indent=2)

    @staticmethod
    def load(file):
        with open(file, "r") as f:
            cfg = yaml.safe_load(f)
        return Config(cfg)

    def flatten(self):
        return expand_keys(self.cfg)

    def dump(self, file):
        with open(file, "w") as f:
            yaml.safe_dump(self.cfg, f, indent=2)

    def digest(self, ignore=None):
        cfg = allbut(self.cfg, ignore)
        cfg_dump = yaml.safe_dump(cfg, sort_keys=True).encode("uft-8")
        return hashlib.md5(cfg_dump).hexdigest()
