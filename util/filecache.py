import copy
import json
import pathlib
import pickle
import yaml
import pandas as pd

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_csv(path):
    return pd.read_csv(path)

def load_txt(path):
    with open(path, 'r') as f:
        return f.read()

Loaders = {
    '.yml': load_yaml,
    '.yaml': load_yaml,
    '.json': load_json,
    '.csv': load_csv,
    '.txt': load_txt,
}

class FileCache:

    def __init__(self, cache_file=None, loaders=None):
        self.cache_file = pathlib.Path(cache_file)
        self.cache = {}
        self.mtime = {}
        self.loaders = loaders if loaders is not None else Loaders
        if self.cache_file is not None and self.cache_file.exists():
            self.load()

    def dump(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump((self.cache, self.mtime), f)

    def load(self):
        with open(self.cache_file, "rb") as f:
            self.cache, self.mtime = pickle.load(f)

    def wipe(self):
        self.cache = {}
        self.mtime = {}
        if self.cache_file is not None:
            self.dump()

    def get(self, file):
        file = pathlib.Path(file).absolute()
        if not file.exists():
            return None
        if file not in self.cache or file.stat().st_mtime != self.mtime[file]:
            self.cache[file] = self.loaders[file.suffix](file)
            self.mtime[file] = file.stat().st_mtime
        return copy.deepcopy(self.cache[file])
