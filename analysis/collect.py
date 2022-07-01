import getpass
import itertools
import json
import pathlib
import collections
from typing import Optional
from fnmatch import fnmatch

from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from ..util.config import HDict, valmap, keymap
from ..util import FileCache

# unused import because we want the patching side-effects
# on pd.DataFrames
from ..pandas import api


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def shorthand_names(names):
    cols = []
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            cols.append("_".join(parts[-i:]))
    counts = collections.Counter(cols)
    column_renames = {}
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            new_name = "_".join(parts[-i:])
            if counts[new_name] == 1:
                column_renames[c] = new_name
                break
    return column_renames


def shorthand_columns(df):
    column_renames = shorthand_names(df.columns)
    df.rename(columns=column_renames, inplace=True)
    return df


class ResultsLoader:
    def __init__(self, cache_file: Optional[str] = None, num_workers: int = 8):
        unixname = getpass.getuser()
        cache_file = cache_file or f"/tmp/{unixname}-results.diskcache"
        self._cache = FileCache(cache_file)
        self._num_workers = num_workers

    def load_configs(
        self,
        *paths,
        shorthand=True,
        properties=False,
        metadata=False,
        log=False,
        callbacks=False,
    ):

        folders = list(
            itertools.chain.from_iterable(
                pathlib.Path(path).iterdir() for path in paths
            )
        )

        configs = self._cache.gets(
            (folder / "config.yml" for folder in folders), num_workers=self._num_workers
        )

        rows = []
        for folder, cfg in tqdm(zip(folders, configs), leave=False, total=len(folders)):
            if cfg is None:
                continue
            cfg = HDict(cfg)
            if not log:
                cfg.pop("log", None)
            if not callbacks:
                cfg.pop("callbacks", None)
            if metadata:
                cfg.update({"metadata": self._cache[folder / "metadata.json"]})
            if properties:
                cfg.update({"properties": self._cache[folder / "properties.json"]})

            flat_cfg = valmap(list2tuple, cfg.flatten())
            flat_cfg["path"] = folder

            flat_cfg = keymap(lambda x: x.replace("._class", ""), flat_cfg)
            flat_cfg = keymap(lambda x: x.replace("._fn", ""), flat_cfg)

            rows.append(flat_cfg)

        df = pd.DataFrame.from_records(rows)

        for col in df.columns:
            if not df[col].map(pd.api.types.is_hashable).all():
                df[col] = df[col].map(json.dumps)

        if shorthand:
            df = shorthand_columns(df)
        return df

    def load_metrics(
        self,
        config_df,
        file="metrics.jsonl",
        prefix="log",
        shorthand=True,
        copy_cols=None,
        path_key=None,
    ):

        log_dfs = []
        if path_key is None:
            path_key = "path"
        if copy_cols is None:
            copy_cols = config_df.columns.to_list()

        folders = config_df[path_key].values

        files = self._cache.gets(
            (folder / file for folder in folders), num_workers=self._num_workers
        )

        for (_, row), log_df in tqdm(
            zip(config_df.iterrows(), files), total=len(config_df), leave=False
        ):
            row = row.to_dict()
            path = pathlib.Path(row[path_key])
            if log_df is None:
                continue
            if prefix:
                log_df.rename(
                    columns={c: f"{prefix}.{c}" for c in log_df.columns}, inplace=True
                )
            if len(copy_cols) > 0:
                for col in copy_cols:
                    val = row[col]
                    if isinstance(val, tuple):
                        log_df[col] = np.array(itertools.repeat(val, len(log_df)))
                    else:
                        log_df[col] = val
            log_df["path"] = path
            log_dfs.append(log_df)

        full_df = pd.concat(log_dfs, ignore_index=True)

        if shorthand:
            renames = {}
            for c in full_df.columns:
                if c.startswith("log."):
                    shortc = c[len("log.") :]
                    if shortc not in full_df.columns:
                        renames[c] = shortc
                    else:
                        renames[c] = c.replace(".", "__")
            full_df.rename(columns=renames, inplace=True)

        return full_df

    def load_aggregate(
        self, config_df, metric_df, agg=None, metrics_groupby=("phase",)
    ):

        config_cols = list(config_df.columns)
        metric_cols = list(set(metric_df.columns) - set(config_df.columns))

        _agg_fns = collections.defaultdict(list)

        DEFAULT_AGGS = {
            "max": ["*acc*", "*score*", "*epoch*"],
            "min": ["*loss*", "*err*"],
        }

        for agg_fn, patterns in DEFAULT_AGGS.items():
            for column in metric_cols:
                if any(fnmatch(column, p) for p in patterns):
                    _agg_fns[column].append(agg_fn)
        if agg is not None:
            for column, agg_fns in agg.items():
                _agg_fns[column].extend(agg)

        g = config_cols + list(metrics_groupby)
        agg_df = metric_df.groupby(g, as_index=False, dropna=False, observed=True).agg(
            _agg_fns
        )
        agg_df.columns = [
            col if agg == "" else f"{agg}_{col}" for col, agg in agg_df.columns.values
        ]

        return agg_df

    def load_all(self, *paths, shorthand=True, **selector):

        dfc = self.load_configs(*paths, shorthand=shorthand,).select(**selector).copy()
        df = self.load_metrics(dfc)
        dfa = self.load_aggregate(dfc, df)
        return dfc, df, dfa
