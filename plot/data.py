from collections import Counter
from contextlib import contextmanager
import pandas as pd


def df_filter(df, **kwargs):

    for k, vs in kwargs.items():
        if not isinstance(vs, list):
            vs = [vs]
        df = df[getattr(df, k).isin(vs)]
        # for v in vs:
        #     selector |= (df == v)
        # df = df[selector]
    return df


def augment_df(df, fns):
    for fn in fns:
        df[fn.__name__] = df.apply(fn, axis=1)
    return df


def unique_combinations(df, cols):
    count = Counter()
    for _, x in df[cols].iterrows():
        count[tuple(x)] += 1
    data = []
    for c, v in count.items():
        data.append(list(c) + [v])
    uniq = pd.DataFrame(data, columns=cols + ["count"]).sort_values(by=cols)
    return uniq


def unique_per_col(df):
    return df.apply(lambda x: set([str(i) for i in x]))


@contextmanager
def inplace_pandas_csv(path):
    df = pd.read_csv(path)
    yield df
    df.to_csv(path, index=False)
