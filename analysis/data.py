def filter_df(df, **kwargs):
    for k, vs in kwargs.items():
        if not isinstance(vs, list):
            df = df[getattr(df, k) == vs]
        else:
            df = df[getattr(df, k).isin(vs)]
    return df


def augment_df(df, fns):
    for fn in fns:
        df[fn.__name__] = df.apply(fn, axis=1)
    return df


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
