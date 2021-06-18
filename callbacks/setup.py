import json

import pandas as pd

from ..util import S3Path
from ..util.summary import summary


def Summary(experiment, filename="summary.txt"):

    x, _ = next(iter(experiment.train_dl))

    # Save model summary
    summary_path = experiment.path / filename

    if not summary_path.exists():

        with summary_path.open("w") as f:
            s = summary(
                experiment.model, x.shape[1:], echo=False, device=experiment.device
            )
            print(s, file=f)

            print("\n\nOptim\n", file=f)
            print(experiment.optim, file=f)

            if experiment.scheduler is not None:
                print("\n\nScheduler\n", file=f)
                print(experiment.scheduler, file=f)

        with summary_path.with_suffix(".json").open("w") as f:
            s = summary(
                experiment.model,
                x.shape[1:],
                echo=False,
                device=experiment.device,
                as_stats=True,
            )
            json.dump(s, f)


def Topology(experiment, filename="topology", extension="pdf"):
    assert extension in ("pdf", "svg")
    from torchviz import make_dot

    x, y = next(iter(experiment.train_dl))
    x = x.to(experiment.device)
    y = y.to(experiment.device)

    # Save model topology
    topology_path = experiment.path / filename
    topology_pdf_path = topology_path.with_suffix("." + extension)

    if not topology_pdf_path.exists():

        yhat = experiment.model(x)
        loss = experiment.loss_func(yhat, y)
        g = make_dot(loss)
        if extension == "svg":
            g.format = "svg"
        with S3Path.as_local(topology_path) as lf:
            g.render(lf)
        # Interested in pdf, the graphviz file can be removed
        if topology_path.exists():
            topology_path.unlink()


def ParameterTable(experiment, save=False):

    from rich.console import Console
    from rich.table import Table

    console = Console(width=150)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Param")
    table.add_column("Shape", justify="right")
    table.add_column("Numel", justify="right")
    table.add_column("Grad")
    table.add_column("Dtype")
    table.add_column("Dev")

    data = []
    for k, v in experiment.model.named_parameters():
        row = [k, tuple(v.size()), v.numel(), v.requires_grad, v.dtype, v.device]
        data.append(row)
        table.add_row(*[str(i) for i in row])

    console.print(table)

    if save:
        columns = ["param", "shape", "numel", "grad", "dtype", "device"]
        df = pd.DataFrame(data, columns=columns)
        with (experiment.path / "params.csv").open('w') as f:
            df.to_csv(f, index=False)


def ModuleTable(experiment, save=False):

    from rich.console import Console
    from rich.table import Table

    console = Console(width=150)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Module")
    table.add_column("Name")

    data = []
    for k, m in experiment.model.named_modules():
        row = [m.__class__.__name__, k]
        data.append(row)
        table.add_row(*[str(i) for i in row])

    console.print(table)

    if save:
        columns = ["module", "name"]
        df = pd.DataFrame(data, columns=columns)
        with (experiment.path / "modules.csv").open('w') as f:
            df.to_csv(f, index=False)
