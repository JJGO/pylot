import numpy as np


def autochoice(options):
    if isinstance(options, list):
        idx = np.random.randint(len(options))
        return options[idx]
    if isinstance(options, dict):
        keys = list(options.keys())
        idx = np.random.randint(len(options))
        key = keys[idx]
        return key, options[key]
