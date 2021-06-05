import functools
import warnings
import numpy as np
from scipy import stats

from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import matplotlib.pyplot as plt
import seaborn as sns

from pylot.util import separate_kwargs
import pylot.pandas

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


def figure_size(row=None, data=None):
    height = 16
    if row is not None:
        height /= data[row].nunique()
    height = min(height, 8)
    width = 10
    aspect = width / height
    return height, aspect


def factor_plot(
    data,
    x,
    y,
    hue,
    col,
    row,
    xlog=False,
    ylog=False,
    xlim=None,
    ylim=None,
    sns_kwargs=None,
    **kwargs,
):
    if sns_kwargs is None:
        sns_kwargs = {}

    _sns_kwargs, kwargs = separate_kwargs(kwargs, sns.relplot)
    sns_kwargs.update(**_sns_kwargs)

    data = data.select(**kwargs)

    assert len(data) > 0, "Empty dataset"

    height, aspect = figure_size(row, data)

    sns_defaults = dict(
        data=data,
        hue=hue,
        col=col,
        row=row,
        col_wrap=None,
        kind="line",
        legend="full",
        palette=viridis_high,
        hue_norm=infer_norm(data[hue].unique()),
        height=height,
        aspect=aspect,
        markers=True,
        # markersize=5,
    )
    # err_style="band",
    fig = sns.relplot(x=x, y=y, **{**sns_defaults, **sns_kwargs})
    if xlog:
        plt.xscale("log")

    if ylog:
        plt.yscale("log")

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    return fig


def cat_plot(
    data,
    x,
    y,
    hue,
    col,
    row,
    xlog=False,
    ylog=False,
    xlim=None,
    ylim=None,
    sns_kwargs=None,
    **kwargs,
):

    if sns_kwargs is None:
        sns_kwargs = {}

    _sns_kwargs, kwargs = separate_kwargs(kwargs, sns.catplot)
    sns_kwargs.update(**_sns_kwargs)

    data = data.select(**kwargs)

    assert len(data) > 0, "Empty dataset"

    height, aspect = figure_size(row, data)

    sns_defaults = dict(
        data=data,
        hue=hue,
        col=col,
        row=row,
        col_wrap=None,
        kind="bar",
        legend="full",
        palette="tab20",
        height=height,
        aspect=aspect,
    )
    fig = sns.catplot(x=x, y=y, **{**sns_defaults, **sns_kwargs})
    if xlog:
        plt.xscale("log")

    if ylog:
        plt.yscale("log")

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    return fig


def factor_plot_df(data, **kwargs):
    return functools.partial(factor_plot, data, **kwargs)


def cat_plot_df(data, **kwargs):
    return functools.partial(cat_plot, data, **kwargs)
