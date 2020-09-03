import collections
import functools
import inspect
import pathlib
import shutil
import numpy as np
import pandas as pd


def filter_df(df, **kwargs):
    attrs = df.attrs
    for k, vs in kwargs.items():
        if not isinstance(vs, list):
            df = df[getattr(df, k) == vs]
        else:
            df = df[getattr(df, k).isin(vs)]
    df.attrs = attrs
    return df


def augment_df(df, fn, name=None):
    name = fn.__name__ if name is None else name
    params = list(inspect.signature(fn).parameters.keys())
    fixed = {p: df.attrs["uniq"][p] for p in params if p not in df.columns}
    params = [p for p in params if p not in fixed]

    if len(fixed) > 0:
        fn = functools.partial(fn, **fixed)

    def wrapper(row):
        kwargs = {k: row.get(k) for k in params}
        return fn(**kwargs)

    df[name] = df.apply(wrapper, axis=1)


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


def move_df(df, root):
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = df.path.values
    for path in paths:
        if not path.exists():
            continue
        target = root / path.name
        # path.replace(target)
        shutil.move(path, target)


def broadcast_attr(df, col, to=None, concat=True, **when):

    tmp = filter_df(df, **when)
    vals = tmp[col].unique()
    if to is None:
        to = [c for c in df[~df[col].isna()][col].unique() if c not in vals]
    if not isinstance(to, (tuple, list)):
        to = [to]
    join = np.array([[old, new] for old in vals for new in to])
    join = pd.DataFrame(data=join, columns=[f"{col}_old", col])
    tmp = tmp.rename(columns={col: f"{col}_old"})
    tmp = pd.merge(tmp, join, on=f"{col}_old")
    tmp.drop(columns=[f"{col}_old"])
    if concat:
        merged = pd.concat([df, tmp])
        merged.attrs = df.attrs
        return merged
    return tmp


def df_from_dict(d):
    return pd.DataFrame.from_dict(d, orient="index", columns=[""])


def unique_per_column(df, every=False, counts=True, pretty=False):
    uniqs = {}
    for c in df.columns:

        x = df[c].unique()
        if not every and len(x) == len(df):
            continue
        if counts:
            counts = collections.Counter(df[c].sort_values().values)
            uniqs[c] = list(counts.items())
        else:
            uniqs[c] = x
    if not pretty:
        return uniqs

    index = list(uniqs.keys())
    data = []
    for c in uniqs:
        if counts:
            data.append(["   ".join([f"{v} ({c})" for v, c in uniqs[c]])])
        else:
            data.append([" ".join(uniqs[c])])
    return pd.DataFrame(data=data, index=index, columns=[""])


def acc2err(df):
    for c in df:
        if "acc" in c:
            c2 = c.replace("acc", "err")
            df[c2] = 100 * (1 - df[c])
            df.attrs["log_cols"].append(c2)
