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

from ..util import MeterCSVLogger, printc
from ..util import dict_recursive_update, expand_dots, allbut, get_full_env_info
from .util import fix_seed


class Experiment:

    DEFAULT_CFG = pathlib.Path("default.yml")
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

        self.csvlogger = None

    def _init_new(self, cfg, uid=None, **kwargs):

        default = {"experiment": {"seed": 42, "type": f"{self.__class__.__name__}"}}
        # 1. Default config
        if Experiment.DEFAULT_CFG.exists():
            with open(Experiment.DEFAULT_CFG, "r") as f:
                default = yaml.load(f, Loader=yaml.FullLoader)
        # 2. cfg dict or file
        if cfg is not None:
            # File
            if isinstance(cfg, (str, pathlib.Path)):
                with open(cfg, "r") as f:
                    cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = dict_recursive_update(default, cfg)
        else:
            cfg = default
        # 3. Forced kwargs
        kwargs = expand_dots(kwargs)
        self.cfg = dict_recursive_update(cfg, kwargs)

        root = pathlib.Path()
        if self.get_param("log.root", False):
            root = pathlib.Path(self.get_param("log.root"))

        if uid is None:
            uid = self.generate_uid()
        self.uid = uid
        self.path = pathlib.Path(root) / self.uid

    def _init_existing(self, path):
        existing_cfg = pathlib.Path(path) / "config.yml"
        assert existing_cfg.exists(), "Cannot find config.yml under the provided path"
        self.path = pathlib.Path(path)
        with open(self.path / "config.yml", "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.uid = self.path.stem

    def save_config(self):
        self.path.mkdir(exist_ok=True, parents=True)
        path = self.path / "config.yml"
        with open(path, "w") as f:
            yaml.dump(self.cfg, f, indent=2)

    @property
    def digest(self):
        cfg = allbut(self.cfg, ("log",))
        return hashlib.md5(yaml.dump(cfg, sort_keys=True).encode("utf-8")).hexdigest()

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
        printc(f"Logging results to {self.path}", color="MAGENTA")

        self.csvlogger = MeterCSVLogger(self.path / "logs.csv")

        # Save environment for repro
        envinfo_path = self.path / "env.yml"
        with open(envinfo_path, "a+") as f:
            yaml.dump([get_full_env_info()], f)
        self.save_config()

    def log(self, *args, **kwargs):
        self.csvlogger.set(*args, **kwargs)

    def dump_logs(self):
        self.csvlogger.set(timestamp=time.time())
        self.csvlogger.flush()

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
