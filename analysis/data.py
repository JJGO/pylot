import collections
import functools
import inspect
import pathlib
import shutil
import numpy as np
import pandas as pd


def unique_combinations(df, cols):
    # Like .unique() but for combinations of columns and with counts as well
    return df.groupby(by=cols).size().reset_index().rename(columns={0: "count"})


def group_mean_std(df):
    drop = []
    for c in df.columns:
        if c.endswith("_mean"):
            drop.append(c)
            prefix = c.split("_mean")[0]
            std = f"{prefix}_std"
            if std in df.columns:
                drop.append(std)
                df[prefix] = df[c] + 1.0j * df[std]
            else:
                df[prefix] = df[c]
    df.drop(columns=drop)
    return df


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


def repivot_loss(df):
    ignore_cols = [c for c in df.columns if not c.endswith("_loss")]
    loss_cols = [c for c in df.columns if c.endswith("_loss")]
    dfp = pd.melt(
        df,
        id_vars=ignore_cols,
        value_vars=loss_cols,
        var_name="phase",
        value_name="loss",
    )
    dfp.attrs["exp_cols"] = df.attrs["exp_cols"] + ["phase"]
    dfp.attrs["log_cols"] = [c for c in df.attrs["log_cols"] if c not in loss_cols] + [
        "loss"
    ]
    # dfp.attrs["uniq"] = df.attrs["uniq"]
    dfp.phase = dfp.phase.str[: -len("_loss")]
    dfp = dfp[~dfp.loss.isna()]
    return dfp


def acc2err(df):
    for c in df:
        if "acc" in c:
            c2 = c.replace("acc", "err")
            df[c2] = 100 * (1 - df[c])
            df.attrs["log_cols"].append(c2)
