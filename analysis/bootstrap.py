from collections import defaultdict
from typing import List

import numpy as np

import pandas as pd

from .aggregate import build_levels, group_data


def bootstrap(df, seed):
    # Given a dataframe sample N items with replacement
    N = len(df)
    rng = np.random.default_rng(seed)
    idxs = rng.integers(N, size=N)
    return df.iloc[idxs]


import functools


def bootstrap_data(data, ablation, seed):
    # Apply bootstrapping at the subject level for each
    # task independently
    fn = functools.partial(bootstrap, seed=seed)
    cols = ablation + ["dataset", "task", "label"]
    return data.groupby(cols).apply(fn).reset_index(drop=True)


def bootstrap_and_group_data(data, ablation, n_boot=100):
    # peform N rounds of bootstrapping, grouping data in the process
    all_dfs = defaultdict(list)
    for seed in range(n_boot):
        data_boot = bootstrap_data(data, ablation, seed)
        data_boot["bootstrap_seed"] = seed
        dfs = group_data(data_boot, ablation)
        dfs.pop("per-subject")
        for k in dfs:
            all_dfs[k].append(dfs[k])

    return {k: pd.concat(vals, ignore_index=True) for k, vals in all_dfs.items()}


def bootstrap_as_std(
    data: pd.DataFrame,
    data_boot: pd.DataFrame,
    ablation: List[str],
    agg_order: List[str],
):
    levels = build_levels(ablation, agg_order)

    for level, (grp_cols, _) in levels.items():
        std = data_boot[level].groupby(grp_cols, as_index=False).std()
        std = std.rename(columns={"dice_score": "dice_score_std_boot"})
        std = std[grp_cols + ["dice_score_std_boot"]]

        data[level] = pd.merge(data[level], std, on=grp_cols)

    return data


def group_with_bootstrapping(
    df: pd.DataFrame, ablation: List[str], agg_order: List[str], n_boot=100
):
    data = group_data(df, ablation, agg_order)
    data_boot = bootstrap_and_group_data(df, ablation, n_boot)
    bootstrap_as_std(data, data_boot, ablation, agg_order)
    for k, v in data_boot.items():
        data[f"{k}-bootstrap"] = v
    return data
