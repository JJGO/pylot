from concurrent.futures import ThreadPoolExecutor


def parallel_mapping_items(mapping, max_workers=8):
    keys = list(mapping)

    def _load(key):
        return key, mapping[key]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(_load, keys)


def parallel_iter_dataset(dataset, max_workers=8):
    keys = range(len(dataset))

    def _load(idx):
        return dataset[idx]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(_load, keys)
