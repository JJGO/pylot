from typing import Any, Callable, Sequence, Union

import pandas as pd


def groupby_mode_nonum(
    df: pd.DataFrame,
    groupby: Sequence[str],  # columns
    number_agg: Union[str, Callable],
    enforce_unique_str: bool = True,
    multiple_token: Any = "MULTIPLE",
    **groupby_kws,
):
    """
    This function's raison-de-etre is that pandas will drop str
    columns when doing a groupby and aggregating using a numerical
    function such as mean or max. This function, will try to keep
    these columns around using the mode
    """
    def str_agg(col):
        if col.nunique() == 1:
            return col.mode()
        if enforce_unique_str:
            raise ValueError(f"Multiple values for col {col}")
        return multiple_token

    agg_by_type = {
        "number": number_agg,
        "object": str_agg,
    }

    cols_by_type = {
        type_: df.select_dtypes(type_).columns.difference(groupby)
        for type_ in agg_by_type
    }

    agg_by_col = {
        col: (col, agg_by_type[type_])
        for type_, cols in cols_by_type.items()
        for col in cols
    }

    return df.groupby(groupby, **groupby_kws).agg(**agg_by_col)
