import os

from datetime import datetime
import pathlib

import matplotlib.pyplot as plt

from IPython.display import HTML, display

# from ..util.meta import delegates
from ..util.jupyter import isnotebook, notebook_put_into_clipboard

import matplotlib.figure
import seaborn.axisgrid
import plotly.graph_objs._figure


# @delegates(to=plt.savefig)
def publishfig(path, fig=None, formats=("svg",), root=None, base_url=None, **kwargs):

    if fig is None:
        fig = plt.gcf()

    if root is None:
        root = os.environ["FIGURE_ROOT"]
    root = pathlib.Path(root)

    if base_url is None:
        base_url = os.environ["FIGURE_URL"]

    path = f"{path}-{datetime.now():%F}"
    link = f"{base_url}/{path}.{formats[0]}"
    full_path = root / path

    if "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = "tight"

    for extension in formats:
        file_path = full_path.with_suffix(f".{extension}")
        file_path.parent.mkdir(exist_ok=True, parents=True)

        if isinstance(fig, matplotlib.figure.Figure):
            fig.savefig(file_path, format=extension, **kwargs)

        elif isinstance(fig, seaborn.axisgrid.FacetGrid):
            g = fig
            g.fig.savefig(file_path, format=extension, **kwargs)

        elif isinstance(fig, plotly.graph_objs._figure.Figure):
            fig.write_image(file_path.as_posix())

        else:
            raise ValueError(f"{type(fig)} not supported")

    if isnotebook():
        notebook_put_into_clipboard(link)
        display(HTML(f"<h3> <a href={link}> {link} </a> </h3>"))

    return link


def publishHTML(path, html, root=None, base_url=None, **kwargs):
    if root is None:
        root = os.environ["FIGURE_ROOT"]
    root = pathlib.Path(root)

    if base_url is None:
        base_url = os.environ["FIGURE_URL"]

    path = f"{path}-{datetime.now():%F}"
    link = f"{base_url}/{path}.html"
    full_path = root / path

    with full_path.with_suffix(".html").open("w") as f:
        f.write(f"<html><body>{html}</body></html>")

    if isnotebook():
        notebook_put_into_clipboard(link)
        display(HTML(f"<h3> <a href={link}> {link} </a> </h3>"))
        display(HTML(html))

    return link


def publishImage(path, image, root=None, base_url=None, **kwargs):
    if root is None:
        root = os.environ["FIGURE_ROOT"]
    root = pathlib.Path(root)

    if base_url is None:
        base_url = os.environ["FIGURE_URL"]

    path = f"{path}-{datetime.now():%F}"
    link = f"{base_url}/{path}.png"
    full_path = root / path
    full_path.parent.mkdir(exist_ok=True, parents=True)

    image.save(str(full_path.with_suffix(".png")))

    if isnotebook():
        notebook_put_into_clipboard(link)
        display(HTML(f"<h3> <a href={link}> {link} </a> </h3>"))
        display(image)

    return link
