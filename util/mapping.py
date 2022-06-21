import collections
import copy
import itertools


def allbut(mapping, keys):
    mapping = copy.deepcopy(mapping)
    for k in keys:
        if k in mapping:
            del mapping[k]
    return mapping


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def dict_recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def delete_with_prefix(d, pre):
    todelete = []
    for k in d:
        if k.startswith(pre):
            todelete.append(k)
    for k in todelete:
        del d[k]
    return d
