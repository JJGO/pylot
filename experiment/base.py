from abc import abstractmethod
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
import pathlib
import random
import shutil
import signal
import string
import sys
import time
import yaml

import pandas as pd

from ..util import dict_recursive_update, expand_dots, allbut, get_full_env_info, make_path, S3Path, printc
from .util import fix_seed


class Experiment:

    NODEFAULT = object()

    def __init__(self, cfg=None, path=None, **kwargs):
        if path is not None:
            assert (
                cfg is None
            ), "Config must not be provided when loading an existing experiment"
            assert (
                kwargs == {}
            ), "Keyword arguments must not be provided when loading an existing experiment"
            self._init_existing(path)
        else:
            self._init_new(cfg, **kwargs)

        fix_seed(self.cfg["experiment"]["seed"])
        # signal.signal(signal.SIGINT, self.SIGINT_handler)
        signal.signal(signal.SIGQUIT, self.SIGQUIT_handler)

    def _init_new(self, cfg, uid=None, **kwargs):

        default = {"experiment": {"seed": 42, "type": f"{self.__class__.__name__}"}}
        # 2. cfg dict or file
        if cfg is not None:
            cfg = dict_recursive_update(default, cfg)
        else:
            cfg = default
        # 3. Forced kwargs
        kwargs = expand_dots(kwargs)
        self.cfg = dict_recursive_update(cfg, kwargs)

        if uid is None:
            uid = self.generate_uid()
        self.uid = uid

        root = self.get_param('log.root', '.')
        root = make_path(root)
        self.path = root / self.uid

    def _init_existing(self, path):
        existing_cfg = pathlib.Path(path) / "config.yml"
        assert existing_cfg.exists(), "Cannot find config.yml under the provided path"
        self.path = pathlib.Path(path)
        with (self.path / 'config.yml').open('r') as f:
            self.cfg = yaml.safe_load(f)
        self.uid = self.path.stem

    def save_config(self):
        self.path.mkdir(exist_ok=True, parents=True)
        path = self.path / "config.yml"
        with path.open('w') as f:
            yaml.safe_dump(self.cfg, f, indent=2)

    @property
    def digest(self):
        cfg = allbut(self.cfg, ("log", "tags"))
        return hashlib.md5(
            yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def __hash__(self):
        return int(self.digest, 16)

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
        nonce = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        return f"{now}-{nonce}-{self.digest}"

    def build_logging(self):
        assert hasattr(self, "uid"), "UID needs to have been generated first"
        assert hasattr(self, "path"), "UID needs to have been generated first"
        self.path.mkdir(exist_ok=True, parents=True)
        # printc(f"Logging results to {self.path}", color="MAGENTA")
        print(f"{self.path}")

        # self.csvlogger = MeterCSVLogger(self.logs_path)
        # self.tensor_store = TensorStore(self.path / "store.hdf5")

        # Save environment for repro
        envinfo_path = self.path / "env.yml"
        if not envinfo_path.exists():
            envinfo_path.touch()
        with envinfo_path.open('a') as f:
            yaml.safe_dump([get_full_env_info()], f)

        self.save_config()

    # def SIGINT_handler(self, signal, frame):
    #     pass

    def SIGQUIT_handler(self, signal, frame):
        self.delete()
        sys.exit(1)

    def delete(self):
        if isinstance(self.path, S3Path):
            self.path.rmtree()
        else:
            shutil.rmtree(self.path, ignore_errors=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cfg})"

    def __str__(self):
        return f"{self.__class__.__name__}\n---\n" + yaml.safe_dump(self.cfg, indent=2)

    @abstractmethod
    def run(self):
        pass

    def resume(self):
        pass

    def load(self, tag):
        pass

    def get_param(self, param, default=NODEFAULT):
        mapping = self.cfg

        for k in param.split("."):
            if k in mapping:
                mapping = mapping[k]
            else:
                if default is self.NODEFAULT:
                    raise LookupError(f"Could find param {param} in config")
                return default

        return mapping

    @property
    def metrics_path(self):
        return self.path / "metrics.jsonl"

    def save_metrics(self, **metrics):

        # Save as JSONL file
        with self.metrics_path.open('a') as f:
            print(json.dumps(metrics), file=f)

    def batch_save_metrics(self, metrics):

        with self.metrics_path.open("a") as f:
            for metric in metrics:
                print(json.dumps(metric), file=f)

    @property
    def metrics(self):
        if self.metrics_path.exists():
            with self.metrics_path.open('r') as f:
                return [json.loads(line) for line in f.readlines()]

    @property
    def metrics_df(self):
        if self.metrics_path.exists():
            with self.metrics_path.open('r') as f:
                return pd.read_json(f, lines=True)

    # LOGS API IS DEPRECATED IN FAVOR OF THE METRICS API

    #     def log(self, *args, **kwargs):
    #         self.csvlogger.set(*args, **kwargs)

    #     def dump_logs(self):
    #         self.csvlogger.set(timestamp=time.time())
    #         self.csvlogger.flush()

    #     @property
    #     def logs_path(self):
    #         return self.path / "logs.csv"

    # @property
    # def logs_df(self):
    #     if self.logs_path.exists():
    #         return pd.read_csv(self.logs_path)
