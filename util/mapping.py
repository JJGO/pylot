import collections
import copy
import itertools
import pandas as pd


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


def expand_dots(d):
    # expand_dots({"a.b.c": 1, "J":2, "a.d":2, "a.b.d":3})
    newd = {}
    for k, v in d.items():
        if "." in k:
            pre, post = k.split(".", maxsplit=1)
            u = expand_dots({post: v})
            if pre in newd:
                newd[pre] = dict_recursive_update(newd[pre], u)
            else:
                newd[pre] = u
        elif isinstance(v, dict):
            u = expand_dots(v)
            if k in newd:
                newd[k] = dict_recursive_update(newd[k], u)
            else:
                newd[k] = u
        else:
            newd[k] = v
    return newd


NODEFAULT = object()


def get_from_dots(mapping, key, default=NODEFAULT):
    for k in key.split("."):
        if k in mapping:
            mapping = mapping[k]
        else:
            if default is NODEFAULT:
                raise LookupError(f"Could find key {key} in config")
            return default

    return mapping


def pop_from_dots(mapping, key, default=NODEFAULT):
    prev = mapping
    for k in key.split("."):
        if k in mapping:
            prev = mapping
            mapping = mapping[k]
        else:
            if default is NODEFAULT:
                raise LookupError(f"Could find key {key} in config")
            return default

    del prev[k]
    return mapping


def allbut(mapping, keys):
    mapping = copy.deepcopy(mapping)
    for k in keys:
        if k in mapping:
            del mapping[k]
    return mapping


def expand_keys(d):
    expanded = {}
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            for k2, v2 in expand_keys(v).items():
                expanded[f"{k}.{k2}"] = v2
        else:
            expanded[k] = v
    return expanded


def delete_with_prefix(d, pre):
    todelete = []
    for k in d:
        if k.startswith(pre):
            todelete.append(k)
    for k in todelete:
        del d[k]
    return d


def dictdiff(*ds, flatten=False):
    if flatten:
        ds = [expand_keys(d) for d in ds]
    from functools import reduce
    import operator

    rows = []
    ks = reduce(operator.or_, [set(d.keys()) for d in ds])
    for k in ks:
        vs = [d.get(k, "") for d in ds]
        if not all(vs[0] == v for v in vs):
            rows.append([k] + vs)
    return pd.DataFrame(data=rows)
