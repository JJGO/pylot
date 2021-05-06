"""Dataset module inclduing preprocessing and custom datasets

The wrappers here include proper
"""

from .util import train_val_split, stratified_train_val_split
from .path import dataset_path, DatapathMixin
from .cache import IndexedImageFolder, IndexedDatasetFolder, IndexedImageDataset
from .subset import subset_dataset
from .selfsuper import SelfsuperDataset, self_supervised
from .vision import (MNIST,
                     CIFAR10,
                     CIFAR100,
                     ImageNet,
                     Places365,
                     TinyImageNet,
                     Miniplaces,
                     nanoImageNet)
