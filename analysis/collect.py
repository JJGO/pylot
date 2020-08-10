from collections import Counter
import itertools
import pathlib

import pandas as pd
from tqdm import tqdm

from ..util import FileCache
from ..util import expand_keys, delete_with_prefix


def shorthand_columns(df, return_dict=False):
    cols = []
    for c in df.columns:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            cols.append("__".join(parts[-i:]))
    counts = Counter(cols)
    column_renames = {}
    for c in df.columns:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            new_name = "__".join(parts[-i:])
            if counts[new_name] == 1:
                column_renames[c] = new_name
                break

    df.rename(columns=column_renames, inplace=True)
    if return_dict:
        return df, column_renames
    return df


def dedup_df(df):
    uniq_cols = []
    for c in df.columns:
        if df[c].nunique() == 1:
            uniq_cols.append([c, df[c].unique()[0]])
    df.drop(columns=[c for c, _ in uniq_cols], inplace=True)
    return pd.DataFrame(data=uniq_cols, columns=["param", "value"])


def list2tuple(val):
    if isinstance(val, list):
        return tuple(val)
    return val


class ResultsLoader:
    def __init__(self, cache_file="/tmp/pylot-results.cache"):
        self.filecache = FileCache(cache_file)

    def load_configs(self, *paths, shorthand=True, dedup=True):
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

        for i, folder in enumerate(tqdm(folders, total=total)):
            cfg = self.filecache.get(folder / "config.yml")
            if cfg is None:
                continue

            cfg = delete_with_prefix(expand_keys(cfg), "log")

            for c in columns:
                val = cfg.get(c, None)
                columns[c].append(list2tuple(val))
            for c in cfg:
                if c not in columns:
                    columns[c] = [None] * i
                    columns[c].append(list2tuple(cfg[c]))
            paths.append(folder)
        columns["experiment.path"] = paths
        df = pd.DataFrame.from_dict(columns)
        self.filecache.dump()
        if shorthand:
            df = shorthand_columns(df)
        if dedup:
            return df, dedup_df(df)
        return df

    def load_logs(self, *paths, shorthand=True, dedup=True):
        df = self.load_configs(*paths, shorthand=False, dedup=dedup)
        if dedup:
            df, uniqs = df

        log_dfs = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            log_df = self.filecache.get(row["experiment.path"] / "logs.csv")
            if log_df is None:
                continue
            log_df.rename(columns={c: f"log.{c}" for c in log_df.columns}, inplace=True)
            for k, v in row.items():
                log_df[k] = [v for _ in range(len(log_df))]
            log_dfs.append(log_df)

        self.filecache.dump()

        full_df = pd.concat(log_dfs, ignore_index=True)
        log_cols = [c for c in full_df.columns if c.startswith("log.")]
        if shorthand:
            full_df, renames = shorthand_columns(full_df, return_dict=True)
            log_cols = [renames[c] for c in log_cols]

        exp_cols = [c for c in full_df.columns if c not in log_cols]

        if dedup:
            return (full_df, uniqs), exp_cols, log_cols
        return full_df, exp_cols, log_cols

    def load_agg_logs(self, *paths, agg=None, shorthand=True, dedup=True):
        df, exp_cols, log_cols = self.load_logs(
            *paths, shorthand=shorthand, dedup=dedup
        )
        if dedup:
            df, uniqs = df

        _agg_fns = {}
        for c in log_cols:
            if c.startswith("t_"):  # Timing, mean
                _agg_fns[c] = "mean"
            elif "loss" in c:  # Min loss
                _agg_fns[c] = ["min", "mean"]
            elif "acc" in c:  # Accuracy, max
                _agg_fns[c] = ["max", "mean"]
            elif "epoch" in c:
                _agg_fns[c] = "max"

        if agg is not None:
            _agg_fns.update(agg)

        g = "path" if shorthand else "experiment.path"
        agg_df = df.groupby(g, as_index=False).agg(_agg_fns)
        agg_df.columns = [
            col if agg == "" else f"{agg}_{col}" for col, agg in agg_df.columns.values
        ]

        exp_df = self.load_configs(*paths, shorthand=shorthand, dedup=dedup)
        if dedup:
            exp_df, _ = exp_df

        exp_cols = list(exp_df.columns)
        log_cols = list(agg_df.columns)
        agg_df = pd.merge(agg_df, exp_df, how="left", on=g)

        if dedup:
            return (agg_df, uniqs), exp_cols, log_cols
        return agg_df, exp_cols, log_cols


def drop_std(df):
    drops = []
    renames = {}
    for c in df.columns:
        if c.endswith("_std"):
            drops.append(c)
        elif c.endswith("_mean"):
            pre = c[: -len("_mean")]
            renames[c] = pre
    df.drop(columns=drops, inplace=True)
    df.rename(columns=renames, inplace=True)
    return df


def merge_std(df):
    pass


# # FIXME DEPRECATED, left for compatibility
# def df_from_results(results_path):
#     columns = {}

#     folders = pathlib.Path(results_path).iterdir()
#     paths = []
#     for i, folder in enumerate(tqdm(folders)):
#         with open(folder / "config.yml", "r") as f:
#             cfg = yaml.load(f, Loader=yaml.FullLoader)

#         ed = delete_with_prefix(expand_keys(cfg), "log")

#         for c in columns:
#             columns[c].append(ed.get(c, None))
#         for c in ed:
#             if c not in columns:
#                 columns[c] = [None] * i
#                 columns[c].append(ed[c])
#         paths.append(folder)
#     columns["experiment.path"] = paths
#     df = pd.DataFrame.from_dict(columns)
#     return df
