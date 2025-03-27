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
Cyberpunk = ListedColormap(
    ["#08F7FE", "#FE53BB", "#F5D300", "#00ff41", "#ff0000", "#9467bd"]
)
plt.register_cmap("Cyberpunk", Cyberpunk)
