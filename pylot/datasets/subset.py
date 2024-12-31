from random import shuffle
from collections import defaultdict
from copy import deepcopy

# Subset a DatasetFolder or a ImageFolder
def subset_dataset(
    dataset, n_classes=None, samples_per_class=None, random=False, inplace=False
):
    if not inplace:
        dataset = deepcopy(dataset)
    # Subset classes
    if n_classes is not None:
        if random:
            shuffle(dataset.classes)
        dataset.classes = dataset.classes[:n_classes]
        dataset.class_to_idx = {
            k: v for k, v in dataset.class_to_idx.items() if k in set(dataset.classes)
        }
        idxs = set(dataset.class_to_idx.values())
        dataset.targets = [i for i in dataset.targets if i in idxs]
        dataset.samples = [i for i in dataset.samples if i[1] in idxs]

    # Subset samples per class
    if random:
        shuffle(dataset.samples)

    count = defaultdict(int)
    new_targets = []
    new_samples = []
    for sample in dataset.samples:
        target = sample[1]
        if count[target] < samples_per_class:
            count[target] += 1
            new_targets.append(target)
            new_samples.append(sample)

    dataset.samples = new_samples
    dataset.targets = new_targets
    return dataset

