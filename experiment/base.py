import collections.abc
import datetime
import hashlib
# import json
import pathlib
import random
import shutil
import signal
import string
import sys
import time

import numpy as np
import torch

from ..util import CSVLogger
from ..util import printc


import yaml


def dict_recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def expand_dots(d):
    # expand_dots({"a.b.c": 1, "J":2, "a.d":2, "a.b.d":3})
    newd = {}
    for k, v in d.items():
        if '.' in k:
            pre, post = k.split('.', maxsplit=1)
            u = expand_dots({post: v})
            if pre in newd:
                newd[pre] = dict_recursive_update(newd[pre], u)
            else:
                newd[pre] = u
        else:
            newd[k] = v
    return newd


class Experiment:

    DEFAULT_CFG = pathlib.Path('default.yml')

    def __init__(self, cfg=None, **kwargs):

        default = {}
        # 1. Default config
        if Experiment.DEFAULT_CFG.exists():
            with open(Experiment.DEFAULT_CFG, 'r') as f:
                default = yaml.load(f, Loader=yaml.FullLoader)
        # 2. cfg dict or file
        if cfg is not None:
            # File
            if isinstance(cfg, (str, pathlib.Path)):
                with open(cfg, 'r') as f:
                    cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = dict_recursive_update(default, cfg)
        else:
            cfg = default
        # 3. Forced kwargs
        kwargs = expand_dots(kwargs)
        cfg = dict_recursive_update(cfg, kwargs)

        if 'experiment' not in cfg:
            cfg['experiment'] = {}
        cfg['experiment']['type'] = f"{self.__class__.__name__}"
        if 'seed' not in cfg['experiment']:
            cfg['experiment']['seed'] = 42

        self.cfg = cfg

        # signal.signal(signal.SIGINT, self.SIGINT_handler)
        signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    def freeze(self):
        self.generate_uid()
        self.fix_seed(self.cfg['experiment']['seed'])
        self.frozen = True

    # def serializable_params(self):
    #     return {k: repr(v) for k, v in self._params.items()}

    def save_config(self):
        path = self.path / 'config.yml'
        with open(path, 'w') as f:
            yaml.dump(self.cfg, f, indent=2)

    def get_path(self):
        ecfg = self.cfg['experiment']
        if 'path' in ecfg:
            return pathlib.Path(ecfg['path'])
        else:
            if 'root' in ecfg:
                parent = pathlib.Path(ecfg['root'])
            else:
                parent = pathlib.Path('results')
            if 'debug' in ecfg and ecfg['debug']:
                parent /= 'tmp'
            parent.mkdir(parents=True, exist_ok=True)
            return parent / self.uid

    @property
    def digest(self):
        return hashlib.md5(yaml.dump(self.cfg, sort_keys=True).encode('utf-8')).hexdigest()

    def __hash__(self):
        return hash(self.digest)

    def fix_seed(self, seed=42, deterministic=False):
        # https://pytorch.org/docs/stable/notes/randomness.html

        # Python
        random.seed(seed)

        # Numpy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def generate_uid(self):
        """Returns a time sortable UID

        Computes timestamp and appends unique identifie

        Returns:
            str -- uid
        """
        if hasattr(self, "uid"):
            return self.uid

        random.seed(time.time())
        N = 4  # length of nonce
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        nonce = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        self.uid = f"{now}-{nonce}-{self.digest}"
        return self.uid

    def build_logging(self):
        self.path = self.get_path()
        printc(f"Logging results to {self.path}", color='MAGENTA')
        self.path.mkdir(exist_ok=True, parents=True)
        self.save_config()

        self.csvlogger = CSVLogger(self.path / 'logs.csv')
        self.log_epoch_n = 0

    def log(self, **kwargs):
        self.csvlogger.set(**kwargs)

    def log_epoch(self, epoch=None):
        if epoch is not None:
            self.log_epoch_n = epoch
        self.log_epoch_n += 1

        self.csvlogger.set(epoch=epoch)
        self.csvlogger.set(timestamp=time.time())
        self.csvlogger.update()
        self.csvlogger.set(epoch=self.log_epoch_n)

    # def SIGINT_handler(self, signal, frame):
    #     pass

    def SIGQUIT_handler(self, signal, frame):
        self.delete()
        sys.exit(1)

    def run(self):
        pass

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cfg})"

    def __str__(self):
        return f"{self.__class__.__name__}\n---\n" + yaml.dump(self.cfg, indent=2)
