import pathlib

import yaml
import json
import pandas as pd
from tqdm import tqdm

from .util import expand_keys, delete_with_prefix


class FileCache:

    def __init__(self, cache_file):
        self.cache_file = pathlib.Path(cache_file)
        self.cache = {}
        if self.cache_file.exists():
            self.load()

    def dump(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def load(self):
        with open(self.cache_file, 'r') as f:
            self.cache = json.load(f)

    def wipe(self):
        self.cache = {}
        self.dump()

    def get(self, file):
        file = pathlib.Path(file).absolute()
        file_str = file.as_posix()
        if file_str not in self.cache:
            if not file.exists():
                content = None
            elif file.suffix in ('.yml', '.yaml'):
                with open(file, 'r') as f:
                    content = yaml.load(f, Loader=yaml.FullLoader)
            elif file.suffix == '.json':
                with open(file, 'r') as f:
                    content = json.load(f)
            self.cache[file_str] = content
        return self.cache[file_str]


# FIXME DEPRECATD, left for compatibility
def df_from_results(results_path):
    columns = {}

    folders = pathlib.Path(results_path).iterdir()
    paths = []
    for i, folder in enumerate(tqdm(folders)):
        with open(folder / 'config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        ed = delete_with_prefix(expand_keys(cfg), 'log')

        for c in columns:
            columns[c].append(ed.get(c, None))
        for c in ed:
            if c not in columns:
                columns[c] = [None] * i
                columns[c].append(ed[c])
        paths.append(folder)
    columns['experiment.path'] = paths
    df = pd.DataFrame.from_dict(columns)
    return df


class ResultsLoader:

    def __init__(self, cache_file='/tmp/filecache'):
        self.filecache = FileCache(cache_file)

    def from_path(self, results_path):
        columns = {}

        folders = pathlib.Path(results_path).iterdir
        total = sum(1 for _ in folders())
        paths = []

        for i, folder in enumerate(tqdm(folders(), total=total)):
            cfg = self.filecache.get(folder / 'config.yml')
            if cfg is None:
                continue

            ed = delete_with_prefix(expand_keys(cfg), 'log')

            for c in columns:
                columns[c].append(ed.get(c, None))
            for c in ed:
                if c not in columns:
                    columns[c] = [None] * i
                    columns[c].append(ed[c])
            paths.append(folder)
        columns['experiment.path'] = paths
        df = pd.DataFrame.from_dict(columns)
        self.filecache.dump()
        return df
