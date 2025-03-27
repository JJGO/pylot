import pathlib
import json
from typing import List, Dict

from collections.abc import Mapping
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

import pandas as pd


class MetricsStore:
    def __init__(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        self.path = path

    def log(self, *metrics: List[Dict[str, Any]]):
        with self.path.open("a") as f:
            for datapoint in metrics:
                print(json.dumps(datapoint), file=f)

    def log_df(self, df: pd.DataFrame):
        with self.path.open("a") as f:
            df.to_json(f, orient="records", lines=True)
            f.write("\n")

    @property
    def data(self) -> List[Dict[str, Any]]:
        if self.path.exists():
            with self.path.open("r") as f:
                return [json.loads(line) for line in f.readlines()]
        return []

    @property
    def df(self) -> Optional[pd.DataFrame]:
        if self.path.exists():
            with self.path.open("r") as f:
                return pd.read_json(f, lines=True)
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.path)})"


class MetricsDict(Mapping):
    def __init__(self, path):
        if isinstance(path, str):
            path = pathlib.Path(path)
        self.path = path

    def __getitem__(self, key):
        return MetricsStore(self.path / f"{key}.jsonl")

    def _find(self):
        return {p.stem: MetricsStore(p) for p in self.path.glob("*.jsonl")}

    def __iter__(self):
        return iter(self._find())

    def __len__(self):
        return len(self._find())

    def __repr__(self):
        return f"{self.__class__.__name__}('{str(self.path)}')"
