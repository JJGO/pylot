import pathlib

import pandas as pd

import pylot.pandas


def phase(metric):
    if "_" in metric:
        return metric.split("_")[0]
    return "train"


def metric(metric):
    if metric.endswith("_std"):
        return None
    if "_" in metric:
        return metric.split("_")[1]
    return metric


def logs2metrics(path):
    path = pathlib.Path(path)
    log_path = path / "logs.csv"
    if not log_path.exists():
        return

    df = pd.read_csv(log_path)
    dfp = pd.melt(df, id_vars=["epoch"], var_name="metric", value_name="value")

    dfp.augment(phase)
    dfp.augment(metric)
    dfp = dfp[~dfp.value.isna()]
    dfp = dfp[~dfp.metric.isna()]
    dfp = dfp[~dfp.epoch.isna()]
    dfq = pd.pivot(
        dfp, index=["epoch", "phase"], columns=["metric"], values="value"
    ).reset_index()
    dfq.to_json(path / "metrics.jsonl", orient="records", lines=True)
    return True
