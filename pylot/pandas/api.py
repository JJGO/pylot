import inspect
import itertools
from typing import Optional, Sequence, Union

import numpy as np
from IPython.display import display as notebook_display

import pandas as pd
from pandas import DataFrame, Series

from .register import register_dataframe_method, register_series_method

_NA = "N/A"


@register_series_method
def fillNA(ser: Series, value, inplace=False) -> Series:
    if not inplace:
        ser = ser.copy()
    if ser.dtype.name == "category":
        ser.cat.add_categories(value, inplace=True)
    ser[ser == _NA] = value
    return ser


@register_series_method
def isNA(ser: Series) -> Series:
    return ser == _NA


@register_series_method
def notNA(ser: Series) -> Series:
    return ser != _NA


@register_series_method
def is_constant(ser: Series) -> bool:
    # A column is "constant" if all the values are the same
    # We consider null values to be different
    return ser.nunique(dropna=False) == 1


@register_dataframe_method
def drop_constant(
    df: DataFrame, inplace=False, ignore: Optional[Sequence[str]] = None
) -> Optional[DataFrame]:

    _ignore = ignore or []
    # Drops columns that are constant
    return df.drop(
        columns=[c for c in df.columns if df[c].is_constant() and c not in _ignore],
        inplace=inplace,
    )


@register_dataframe_method
def constants(df: DataFrame, as_series=False) -> Union[Series, DataFrame]:
    # Returns a dataframe with columns that are constant (all values are the same)
    # and the unique value for each column
    index, vals = [], []
    for c in df:
        if df[c].is_constant():
            vals.append(df[c].unique()[0])
            index.append(c)
    ser = pd.Series(data=vals, index=index)
    if as_series:
        return ser
    return pd.DataFrame(ser, columns=["Unique"])


@register_dataframe_method
def to_categories(df: DataFrame, threshold: float = 0.1, inplace=False) -> DataFrame:
    if not inplace:
        df = df.copy()
    for c in df.select_dtypes(object):
        # Set a threshold in number of categories, by default an order of magnitude less
        if df[c].nunique(dropna=False) < threshold * len(df):
            df[c] = df[c].astype("category")
            if df[c].isna().sum() > 0:
                df[c] = df[c].cat.add_categories(_NA).fillna(_NA)
    return df


@register_dataframe_method
def select(df: DataFrame, **kwargs):
    attrs = df.attrs
    for k, vs in kwargs.items():
        if vs is None:
            df = df[df[k].isna()]
        elif isinstance(vs, list):
            df = df[df[k].isin(vs)]
        else:
            df = df[df[k] == vs]
    df.attrs = attrs
    return df


@register_dataframe_method
def augment(df: DataFrame, fn, name=None, inplace=True) -> Optional[Series]:
    name = name if name else fn.__name__
    params = list(inspect.signature(fn).parameters.keys())
    for param in params:
        assert (
            param in df.columns
        ), f"Function argument '{param}' not in dataframe columns"

    def wrapper(row):
        kwargs = {k: row.get(k) for k in params}
        return fn(**kwargs)

    ser = df.apply(wrapper, axis=1)
    if not inplace:
        return ser
    df[name] = ser


@register_dataframe_method
def broadcast(df: DataFrame, col, to=None, concat=True, **when):
    # Idea is to broadcast an attribute by copying rows and changing the values of a given column
    # This is useful when the value of a metric does not change with respect a experimental condition
    # but in order to plot/analyze is more convenient to add dummy datapoints
    tmp = df.select(**when)
    from_vals = tmp[col].unique()
    if to is None:
        # Unless otherwise specified broadcast to all other present values
        to = [x for x in df[col].unique() if x not in from_vals]
    if not isinstance(to, (tuple, list)):
        to = [to]

    join = np.array([[old, new] for old in from_vals for new in to])
    col_old = col + "_old"
    join = pd.DataFrame(data=join, columns=[col_old, col])
    tmp = tmp.rename(columns={col: col_old})
    tmp = pd.merge(tmp, join, on=col_old)
    tmp.drop(columns=[col_old], inplace=True)
    if concat:
        merged = pd.concat([df, tmp])
        return merged
    return tmp


@register_dataframe_method
def unique_per_col(df: DataFrame, threshold: float = 0.5, constant=False, display=True):
    low = 0 if constant else 1
    uniqs = []
    with pd.option_context("display.max_colwidth", 400):
        for col in df:
            if low < df[col].nunique(dropna=False) < threshold * len(df):
                ser = df[col]
                if ser.isna().any():
                    ser = ser.fillna(_NA)
                ser = ser.value_counts()
                uniqs.append(ser)
                if display:
                    notebook_display(DataFrame(ser).T)
    if not display:
        return uniqs


def df_from_dict(d):
    return DataFrame.from_dict(d, orient="index", columns=[""])


@register_dataframe_method
def constants_subset(df: DataFrame, **selector):
    # Returns variables that are constant in the selected subset
    # while leaving out aspects that are constant for the entire
    # dataset
    sub_df = df.select(**selector)
    sub_constants = sub_df.constants()
    return sub_constants[~sub_constants.index.isin(set(df.constants().index))]


@register_dataframe_method
def smooth(df, column, *, group_by=("path", "phase"), alpha=None, window=None):
    if alpha is not None:
        return df.groupby(list(group_by))[column].transform(
            lambda x: x.ewm(alpha=alpha).mean()
        )
    elif window is not None:
        return df.groupby(list(group_by))[column].transform(
            lambda x: x.rolling(window, center=True).mean()
        )
    raise ValueError("Either alpha or window must be specified")


@register_dataframe_method
def augment_from_attrs(df, prefix=""):
    for col, val in df.attrs.items():
        if isinstance(val, (tuple, list, dict)):
            df[prefix + col] = np.array(itertools.repeat(val, len(df)))
        else:
            df[prefix + col] = val
    return df
