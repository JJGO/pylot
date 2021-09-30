from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import json

import pandas as pd


class MetricsStore:
    def __init__(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        self.path = path

    def dump(self, metrics: Dict[str, Any], batch: bool = False):
        if not batch:
            # Save as JSONL file
            with self.path.open("a") as f:
                print(json.dumps(metrics), file=f)
        else:
            with self.path.open("a") as f:
                for metric in metrics:
                    print(json.dumps(metric), file=f)

    def dump_df(self, df: pd.DataFrame):
        with self.path.open("a") as f:
            df.to_json(f, orient="records", lines=True)

    @property
    def data(self) -> List[Dict[str, Any]]:
        if self.path.exists():
            with self.path.open("r") as f:
                return [json.loads(line) for line in f.readlines()]

    @property
    def df(self) -> pd.DataFrame:
        if self.path.exists():
            with self.path.open("r") as f:
                return pd.read_json(f, lines=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.path)})"


class MetricsDict:
    def __init__(self, path):
        self.path = path
        self.mapping = {}

    def __getitem__(self, key):
        if key not in self.mapping:
            self.mapping[key] = MetricsStore(self.path / f"{key}.jsonl")
        return self.mapping[key]

    def all(self):
        return {p.stem: MetricsStore(p) for p in self.path.glob("*.jsonl")}
