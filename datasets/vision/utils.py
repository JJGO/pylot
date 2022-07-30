from ...util.meter import StatsMeter
import numpy as np
from tqdm.auto import tqdm


def channel_stats(dataset, equal_size=True, precision=None):

    if equal_size:
        meter = StatsMeter()
        # For each image we add a datum (H,W,C) and reduce at the end
        for x, _ in tqdm(dataset):
            meter.add(np.array(x))
        reduction = StatsMeter()
        H, W, C = np.array(x).shape
        N = len(dataset)
        for i in range(H):
            for j in range(W):
                reduction += StatsMeter.from_values(
                    n=N, mean=meter.mean[i, j], std=meter.std[i, j]
                )
        mean, std = reduction.mean, reduction.std

    else:
        # For each image we add data for all the pixels (H*W)
        meter = StatsMeter()
        for x, _ in tqdm(dataset):
            x = np.array(x)
            H, W, C = x.shape
            x = x.reshape(-1, C)
            mu = x.mean(axis=0)
            std = x.std(axis=0)
            meter += StatsMeter.from_values(n=H * W, mean=mu, std=std)
        mean, std = meter.mean, meter.std

    mean = tuple((mean / 255).tolist())
    std = tuple((std / 255).tolist())
    if precision:
        mean = tuple(round(x, precision) for x in mean)
        std = tuple(round(x, precision) for x in std)
    return mean, std
