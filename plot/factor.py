import functools
import warnings
import numpy as np
from scipy import stats

from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import matplotlib.pyplot as plt
import seaborn as sns

from pylot.analysis import filter_df

viridis_high = ListedColormap(cm.get_cmap("viridis").colors[64:])


def infer_norm(array):
    if array.dtype == "O":
        return None
    mean = np.mean(array)
    gmean = stats.gmean(array[np.nonzero(array)[0]])
    median = np.median(array)

    if np.abs(mean - median) > np.abs(gmean - median):
        return LogNorm()
    return None


def factor_plot(
    data,
    x,
    y,
    hue,
    col,
    row,
    style,
    xlog=False,
    ylog=False,
    palette=viridis_high,
    col_wrap=None,
    **kwargs
):
    data = filter_df(data, **kwargs)
    if len(data) == 0:
        warnings.warn("Empty dataset")
        return
    height = 16
    if row is not None:
        height /= data[row].nunique()
    height = min(height, 8)
    width = 10
    aspect = width / height
    fig = sns.relplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        kind="line",
        legend="full",
        palette=palette,
        col=col,
        row=row,
        style=style,
        hue_norm=infer_norm(data[hue].unique()),
        markevery=1,
        markers=True,
        height=height,
        aspect=aspect,
        markersize=5,
        col_wrap=col_wrap,
        err_style="band",
    )
    if xlog:
        plt.xscale("log")

    if ylog:
        plt.yscale("log")

    return fig


def factor_plot_df(data, palette=viridis_high, col_wrap=None):
    return functools.partial(factor_plot, data, palette=palette, col_wrap=col_wrap)
