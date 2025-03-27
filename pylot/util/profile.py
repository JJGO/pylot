import itertools
from typing import Tuple, List

import torch
from tqdm.auto import tqdm

import pandas as pd

from .timer import StatsTimer
from .device import to_device


def benchmark_dataloader(
    dataset: torch.utils.data.Dataset,
    workers: List[int],
    batch_sizes: List[int],
    samples=None,
    skip=5,
    warmup=True,
    dataloader_kws=dict(),
) -> pd.DataFrame:
    # Warm-up data for filesystem cache
    if warmup:
        for _ in tqdm(dataset, leave=False):
            pass
    rows = []

    for num_workers, batch_size in tqdm(
        itertools.product(workers, batch_sizes),
        total=len(batch_sizes) * len(workers),
        leave=False,
    ):
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, **dataloader_kws
        )

        if samples is None:
            steps = len(dl)
        else:
            steps = samples

        timer = StatsTimer(skip=skip)
        it = iter(dl)
        for _ in range(steps):
            with timer("dl"):
                _ = next(it)
        it = iter(dl)
        for _ in range(steps):
            with timer("dl"):
                _ = next(it)

        ms = timer.measurements
        rows.append(
            dict(
                num_workers=num_workers,
                batch_size=batch_size,
                mtime=ms["dl"].mean,
                stime=ms["dl"].std,
            )
        )

    return pd.DataFrame.from_records(rows)


def benchmark_model_speed(
    model: torch.nn.Module,
    input_size: Tuple[int, ...],
    output_size: List[int],
    batch_sizes: List[int],
    loss_func,
    device="cpu",
    samples=100,
) -> pd.DataFrame:

    rows = []
    if samples is None:
        samples = float("inf")

    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    for batch_size in tqdm(batch_sizes, leave=False):

        timer = StatsTimer(skip=5)

        x = torch.zeros((batch_size,) + tuple(input_size))
        y = torch.zeros((batch_size,) + tuple(output_size))

        for _ in range(samples):
            with timer("total"):
                with timer("device"):
                    x, y = to_device([x, y], device)
                with timer("forward"):
                    yh = model(x)
                with timer("backward"):
                    loss_func(yh, y).backward()
                with timer("optim"):
                    optim.step()
                    optim.zero_grad()

        for label, meter in timer.measurements.items():
            rows.append(
                {
                    "batch_size": batch_size,
                    "mtime": meter.mean,
                    "stime": meter.std,
                    "label": label,
                }
            )

    return pd.DataFrame.from_records(rows)
