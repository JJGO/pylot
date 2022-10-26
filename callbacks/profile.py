import itertools
import math

from tabulate import tabulate

from ..util import printc
from ..util.timer import StatsTimer, StatsCUDATimer
from ..util.torchutils import to_device


def Throughput(experiment, cuda=False, n_iter=100):
    """
    Callback to measure time taken by data loader and training loop of the experiment.
    """

    if cuda:
        timer = StatsCUDATimer(unit="ms")
    else:
        timer = StatsTimer(unit="ms")

    # Dataloader throughput
    dl = iter(itertools.cycle(experiment.train_dl))
    for _ in range(n_iter):
        with timer("train-dl"):
            _ = next(dl)

    # Model training throughput
    sample_input = to_device(next(iter(experiment.train_dl)), experiment.device)
    for _ in range(n_iter):
        with timer("train-loop"):
            experiment.run_step(batch=sample_input, batch_idx=0)

    timer_df = timer.measurements_df()
    timer_df = timer_df.set_index("label")

    print(tabulate(timer_df, headers="keys"), flush=True)

    t_dl = timer["train-dl"].mean
    t_gpu = timer["train-loop"].mean
    if t_dl > t_gpu:
        printc(
            f"Experiment is dataloader bound dl={t_dl:.2f}ms > gpu={t_gpu:.2f}ms",
            color="RED",
        )
        recommended_num_workers = math.ceil(
            experiment.train_dl.num_workers * t_dl / t_gpu
        )
        printc(f"Try setting num_workers={recommended_num_workers}", color="RED")
