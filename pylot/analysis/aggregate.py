from typing import Any, Dict, List, Tuple

import pandas as pd

from ..pandas import groupby_mode_nonum


def assert_constants(dataframe: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:

    should_be_constants = columns

    for col in should_be_constants:
        assert dataframe[col].is_constant(), f"column '{col}' is not constant"

    d = dataframe.constants()["Unique"].to_dict()
    constants = {k: d[k] for k in should_be_constants}

    return constants


def build_levels(ablation: List[str], agg_order: List[str]):
    levels = {}
    n = len(agg_order)
    for i in range(n, 0, -1):
        col = agg_order[i - 1]
        levels[f"per-{col}"] = (ablation + agg_order[:i], agg_order[i : i + 1])
    levels["per-ablation"] = (ablation, agg_order[0:1])
    return levels


def group_data(
    df: pd.DataFrame, ablation: List[str], agg_order: List[str], agg: str = "mean"
) -> Dict[str, pd.DataFrame]:
    levels = build_levels(ablation, agg_order)

    dfs = {}

    for level, (grp_cols, drop_cols) in levels.items():
        if drop_cols == []:
            dfs[level] = df
            continue

        df = groupby_mode_nonum(
            df.drop(columns=drop_cols), grp_cols, agg, as_index=False
        )
        dfs[level] = df

    return dfs


def ensure_ablation(
    dfc: pd.DataFrame,
    ablation: List[str],
    not_constant: Tuple[str] = ("path", "seed", "epoch"),
):
    not_constant = list(not_constant) + ablation
    should_be_constants = dfc.columns.difference(not_constant).tolist()
    constants = assert_constants(dfc, should_be_constants)
    dfc.drop_constant(inplace=True, ignore=not_constant)
    return constants
