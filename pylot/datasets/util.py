import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import random_split, Subset


def train_val_split(dataset, val_split, seed=None):
    dataset_size = len(dataset)
    val_size = math.ceil(val_split * dataset_size)
    train_size = dataset_size - val_size
    kwargs = {}
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], **kwargs)


def stratified_train_val_split(dataset, val_split, seed=None):
    indices = np.arange(len(dataset))
    stratify = getattr(dataset, 'targets', None)
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=stratify
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

