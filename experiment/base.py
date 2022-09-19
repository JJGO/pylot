from abc import abstractmethod

import pathlib

import yaml

from .util import fix_seed, absolute_import, generate_tuid

from ..util.metrics import MetricsDict
from ..util.config import HDict, FHDict, ImmutableConfig, config_digest
from ..util.ioutil import autosave
from ..util.libcheck import check_environment


def eval_callbacks(all_callbacks, experiment):
    evaluated_callbacks = {}
    for group, callbacks in all_callbacks.items():
        evaluated_callbacks[group] = []

        for callback in callbacks:
            if isinstance(callback, str):
                cb = absolute_import(callback)(experiment)
            elif isinstance(callback, dict):
                assert len(callback) == 1, "Callback must have length 1"
                callback, kwargs = next(iter(callback.items()))
                cb = absolute_import(callback)(experiment, **kwargs)
            else:
                raise TypeError("Callback must be either str or dict")
            evaluated_callbacks[group].append(cb)
    return evaluated_callbacks


class BaseExperiment:
    def __init__(self, path):
        if isinstance(path, str):
            path = pathlib.Path(path)
        self.path = path
        assert path.exists()
        self.name = self.path.stem

        self.config = ImmutableConfig.from_file(path / "config.yml")
        self.properties = FHDict(self.path / "properties.json")
        self.metadata = FHDict(self.path / "metadata.json")
        self.metricsd = MetricsDict(self.path)

        fix_seed(self.config.get("experiment.seed", 42))
        check_environment()

        self.properties["experiment.class"] = self.__class__.__name__

        if "log.properties" in self.config:
            self.properties.update(self.config["log.properties"])

    @classmethod
    def from_config(cls, config) -> "BaseExperiment":
        if isinstance(config, HDict):
            config = config.to_dict()

        root = pathlib.Path()
        if "log" in config:
            root = pathlib.Path(config["log"].get("root", "."))
        create_time, nonce = generate_tuid()
        digest = config_digest(config)
        uuid = f"{create_time}-{nonce}-{digest}"
        path = root / uuid
        metadata = {"create_time": create_time, "nonce": nonce, "digest": digest}
        autosave(metadata, path / "metadata.json")

        autosave(config, path / "config.yml")
        return cls(str(path.absolute()))

    @property
    def metrics(self):
        return self.metricsd["metrics"]

    def __hash__(self):
        return hash(self.path)

    @abstractmethod
    def run(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}("{str(self.path)}")'

    def __str__(self):
        s = f"{repr(self)}\n---\n"
        s += yaml.safe_dump(self.config._data, indent=2)
        return s

    def build_callbacks(self):
        self.callbacks = {}
        if "callbacks" in self.config:
            self.callbacks = eval_callbacks(self.config["callbacks"], self)
