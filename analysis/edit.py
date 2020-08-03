from contextlib import contextmanager

import json
import yaml
import pandas as pd


@contextmanager
def inplace_pandas_csv(path):
    df = pd.read_csv(path)
    yield df
    df.to_csv(path, index=False)


@contextmanager
def inplace_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    yield data
    with open(path, "w") as f:
        json.dump(data, f)


@contextmanager
def inplace_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    yield data
    with open(path, "w") as f:
        yaml.dump(data, f)
