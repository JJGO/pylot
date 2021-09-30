from contextlib import contextmanager

import json
import shutil
import yaml
import pandas as pd


def _backup_file(path):
    shutil.copy(path, path.with_suffix(path.suffix + ".bk"))


@contextmanager
def inplace_pandas_csv(path, backup=False):
    df = pd.read_csv(path)
    yield df
    if backup:
        _backup_file(path)
    df.to_csv(path, index=False)


@contextmanager
def inplace_json(path, backup=False):
    with open(path, "r") as f:
        data = json.load(f)
    yield data
    if backup:
        _backup_file(path)
    with open(path, "w") as f:
        json.dump(data, f)


@contextmanager
def inplace_yaml(path, backup=False):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    yield data
    if backup:
        _backup_file(path)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, indent=2)


@contextmanager
def inplace_jsonl(path, backup=False):
    df = pd.read_json(path, lines=True)
    yield df
    if backup:
        _backup_file(path)
    df.to_json(path, orient='records', lines=True)
