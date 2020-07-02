import matplotlib.pyplot as plt

from .data import *
from .plot import plot_df
from ..util import AutoMap

CMAP = plt.get_cmap('Set1')
colors = AutoMap(CMAP.colors)

