import json

import pandas as pd
from tabulate import tabulate

from ..util import S3Path
from ..util.summary import summary

from ..metrics import module_table, parameter_table


def ParameterTable(experiment, save=True, verbose=True):

    df = parameter_table(experiment.model)
    if verbose:
        print(tabulate(df, headers="keys"))

    if save:
        with (experiment.path / "params.csv").open("w") as f:
            df.to_csv(f, index=False)


def ModuleTable(experiment, save=True, verbose=True):

    df = module_table(experiment.model)
    if verbose:
        print(tabulate(df, headers="keys"))

    if save:
        with (experiment.path / "modules.csv").open("w") as f:
            df.to_csv(f, index=False)


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


def Topology(experiment, filename="topology", extension="svg"):
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


def CheckHalfCosineSchedule(experiment):

    scheduler = experiment.get_param("train.scheduler.scheduler", None)
    if scheduler == "CosineAnnealingLR":
        T_max = experiment.get_param("train.scheduler.T_max", -1)
        epochs = experiment.get_param("train.epochs")
        assert T_max == epochs, f"T_max not equal to epochs {T_max} != {epochs}"
