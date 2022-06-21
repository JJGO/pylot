from collections import Counter
import itertools
import pathlib

import pandas as pd
from tqdm.autonotebook import tqdm

from .data import drop_std
from ..util import FileCache
from ..util import expand_keys, delete_with_prefix


def shorthand_names(names):
    cols = []
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            cols.append("_".join(parts[-i:]))
    counts = Counter(cols)
    column_renames = {}
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            new_name = "_".join(parts[-i:])
            if counts[new_name] == 1:
                column_renames[c] = new_name
                break
    return column_renames


def shorthand_columns(df, return_dict=False):
    column_renames = shorthand_names(df.columns)
    df.rename(columns=column_renames, inplace=True)
    if return_dict:
        return df, column_renames
    return df


def add_experiment_metadata(df):

    path = df["experiment.path"]
    df["experiment.hash"] = path.map(lambda p: p.name.split("-")[3])
    df["experiment.nonce"] = path.map(lambda p: p.name.split("-")[2])
    df["experiment.create_time"] = path.map(lambda p: p.name[: len("YYYYMMDD-hhmmss")])
    return df


def dedup_df(df):
    uniq = {}
    for c in df.columns:
        if df[c].nunique(dropna=False) == 1:
            uniq[c] = df[c].unique()[0]
    df.drop(columns=list(uniq.keys()), inplace=True)
    return uniq


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


class ResultsLoader:
    def __init__(self, cache_file="/tmp/pylot-results.cache"):
        self.filecache = FileCache(cache_file)
        self.load_metrics = self.load_logs

    def load_configs(
        self, *paths, shorthand=True, dedup=False, metadata=False, categories=False
    ):
        columns = {}

        # folders = pathlib.Path(path).iterdir
        folders = itertools.chain.from_iterable(
            pathlib.Path(path).iterdir() for path in paths
        )
        total = sum(1 for _ in folders)
        folders = itertools.chain.from_iterable(
            pathlib.Path(path).iterdir() for path in paths
        )
        paths = []

        i = 0
        for folder in tqdm(folders, total=total, leave=False):
            cfg = self.filecache.get(folder / "config.yml")
            if cfg is None:
                continue

            # cfg = delete_with_prefix(expand_keys(cfg), "log")
            cfg.pop("log", None)
            cfg = expand_keys(cfg)

            for c in columns:
                val = cfg.get(c, None)
                columns[c].append(list2tuple(val))
            for c in cfg:
                if c not in columns:
                    columns[c] = [None] * i
                    columns[c].append(list2tuple(cfg[c]))

            paths.append(folder)
            i += 1

        columns["experiment.path"] = paths
        df = pd.DataFrame.from_dict(columns)

        if metadata:
            add_experiment_metadata(df)
        self.filecache.dump()
        if shorthand:
            df = shorthand_columns(df)

        if dedup:
            df.attrs["uniq"] = dedup_df(df)
        if categories:
            df = df.to_categories()
        return df

    def load_logs(
        self,
        *paths,
        shorthand=True,
        dedup=False,
        metadata=False,
        df=None,
        file="metrics",
    ):
        if len(paths) > 0:
            df = self.load_configs(
                *paths, shorthand=True, dedup=dedup, metadata=metadata
            )

        log_dfs = []

        for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
            path = row["path"] if shorthand else row["experiment.path"]
            log_df = self.filecache.get(path / f"{file}.jsonl")
            if log_df is None:
                continue

            log_df.rename(columns={c: f"log.{c}" for c in log_df.columns}, inplace=True)
            log_df["path"] = path

            # for k, v in row.items():
            #     log_df[k] = [v for _ in range(len(log_df))]
            log_dfs.append(log_df)

        self.filecache.dump()

        full_df = pd.concat(log_dfs, ignore_index=True)
        full_df = pd.merge(df, full_df, on="path")

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

        full_df.attrs["exp_cols"] = list(df.columns)
        full_df.attrs["log_cols"] = [c for c in full_df.columns if c not in df.columns]
        if dedup:
            full_df.attrs["uniq"] = df.attrs["uniq"]

        return full_df

    def load_parquets(
        self, file, df, prefix=None, shorthand=True, skip=False, copy_cols=("path",),
    ):

        dfxs = []

        for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
            path = row["path"] if shorthand else row["experiment.path"]
            if skip and not (path / f"{file}.parquet").exists():
                continue
            dfx = pd.read_parquet(path / f"{file}.parquet")

            if prefix:
                dfx.rename(
                    columns={c: f"{prefix}.{c}" for c in dfx.columns}, inplace=True
                )

            if copy_cols is not None:
                for c in copy_cols:
                    v = row[c]
                    if v is None or isinstance(v, (str, float, int, pathlib.Path)):
                        dfx[c] = v
                    else:
                        dfx[c] = [v for _ in range(len(dfx))]

            # for k, v in row.items():
            #     if v is None or isinstance(v, (str, float, int)):
            #         dfx[k] = v
            # dfx[k] = [v for _ in range(len(dfx))]

            # Merge is slow
            # dfx = pd.merge(
            #     row.to_frame().transpose(), dfx, on="path", suffixes=("", "_data")
            # )
            dfxs.append(dfx)

        full_df = pd.concat(dfxs, ignore_index=True)
        # full_df = pd.merge(df, full_df, on='path', suffixes=('', "_data"))

        return full_df

    def load_agg_logs(
        self, *paths, agg=None, shorthand=True, dedup=False, metadata=False, df=None
    ):
        if len(paths) > 0:
            df = self.load_logs(
                *paths, shorthand=shorthand, dedup=dedup, metadata=metadata
            )

        _agg_fns = {}
        for c in df.attrs["log_cols"]:
            if c.startswith("t_"):  # Timing, mean
                _agg_fns[c] = "mean"
            elif "loss" in c:  # Min loss
                _agg_fns[c] = ["min"]
            elif "acc" in c:  # Accuracy, max
                _agg_fns[c] = ["max"]
            elif "err" in c:  # Error, min
                _agg_fns[c] = ["min"]
            elif "epoch" in c:
                _agg_fns[c] = "max"
            elif "score" in c:
                _agg_fns[c] = ["max"]

        if agg is not None:
            _agg_fns.update(agg)

        # g = "path" if shorthand else "experiment.path"
        g = df.attrs["exp_cols"] + ["phase"] if "phase" in df.attrs["log_cols"] else []
        agg_df = df.groupby(g, as_index=False, dropna=False, observed=True).agg(
            _agg_fns
        )
        agg_df.columns = [
            col if agg == "" else f"{agg}_{col}" for col, agg in agg_df.columns.values
        ]

        # agg_df = pd.merge(agg_df, exp_df, how="left", on=g)

        agg_df.attrs = df.attrs
        agg_df.attrs["log_cols"] = [c for c in agg_df.columns if c not in g]
        return agg_df

    def load_all(
        self,
        *paths,
        shorthand=True,
        dedup=False,
        metadata=False,
        categories=False,
        **selector,
    ):
        dfc = self.load_configs(
            *paths,
            shorthand=shorthand,
            dedup=dedup,
            metadata=metadata,
            categories=categories,
        )
        from .. import pandas

        dfc = dfc.select(**selector)
        df = self.load_logs(df=dfc)
        dfa = self.load_agg_logs(df=df)

        return dfc, df, dfa
