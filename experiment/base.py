from abc import abstractmethod
import datetime
import hashlib
import pathlib
import random
import shutil
import signal
import string
import sys
import time
import yaml

import numpy as np
import pandas as pd
import torch

from ..util import MeterCSVLogger, printc
from .util import dict_recursive_update, expand_dots


class Experiment:

    DEFAULT_CFG = pathlib.Path('default.yml')

    def __init__(self, cfg=None, path=None, **kwargs):
        if path is not None:
            assert cfg is None, "Config must not be provided when loading an existing experiment"
            assert kwargs == {}, "Keyword arguments must not be provided when loading an existing experiment"
            self._init_existing(path)
        else:
            self._init_new(cfg, **kwargs)

        self.fix_seed(self.cfg['experiment']['seed'])
        # signal.signal(signal.SIGINT, self.SIGINT_handler)
        signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    def _init_new(self, cfg, **kwargs):

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

        if 'root' in cfg:
            root = pathlib.Path(cfg['root'])
            del cfg['root']
        else:
            root = pathlib.Path()

        # Keep track of experiment properties like type, seed, ...
        if 'experiment' not in cfg:
            cfg['experiment'] = {}
        cfg['experiment']['type'] = f"{self.__class__.__name__}"
        if 'seed' not in cfg['experiment']:
            cfg['experiment']['seed'] = 42

        self.cfg = cfg
        self.generate_uid()
        self.path = pathlib.Path(root) / self.uid
        self.path.mkdir(exist_ok=True, parents=True)
        self.save_config()

    def _init_existing(self, path):
        print(path)
        existing_cfg = pathlib.Path(path) / 'config.yml'
        assert existing_cfg.exists(), "Cannot find config.yml under the provided path"
        self.path = pathlib.Path(path)
        with open(self.path / 'config.yml', 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.uid = self.path.stem

    def save_config(self):
        path = self.path / 'config.yml'
        with open(path, 'w') as f:
            yaml.dump(self.cfg, f, indent=2)

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

        Computes timestamp and appends unique identifier

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
        assert hasattr(self, "uid"), "UID needs to have been generated first"
        assert hasattr(self, "path"), "UID needs to have been generated first"
        printc(f"Logging results to {self.path}", color='MAGENTA')

        self.csvlogger = MeterCSVLogger(self.path / 'logs.csv')
        self.csvlogger.set(epoch=None)
        self.log_epoch_n = 0

    def log(self, *args, **kwargs):
        self.csvlogger.set(*args, **kwargs)

    def log_epoch(self, epoch=None):
        if epoch is not None:
            self.log_epoch_n = epoch
        self.log_epoch_n += 1

        self.csvlogger.set(epoch=epoch)
        self.csvlogger.set(timestamp=time.time())
        self.csvlogger.flush()
        self.csvlogger.set(epoch=self.log_epoch_n)

    # def SIGINT_handler(self, signal, frame):
    #     pass

    def SIGQUIT_handler(self, signal, frame):
        self.delete()
        sys.exit(1)

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cfg})"

    def __str__(self):
        return f"{self.__class__.__name__}\n---\n" + yaml.dump(self.cfg, indent=2)

    @abstractmethod
    def run(self):
        pass

    def resume(self):
        pass

    def load(self):
        pass
