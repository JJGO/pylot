import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

Monokai = ListedColormap(
    [
        "#66D9EF",
        "#F92672",
        "#A6E22E",
        "#FD971F",
        "#AE81FF",
        "#ffd866",
        "#DDDDDD",
        "#67a9b5",
        "#d3201f",
        "#47840e",
    ]
)

plt.register_cmap("Monokai", Monokai)
