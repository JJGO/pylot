import collections
import copy


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
        if '.' in k:
            pre, post = k.split('.', maxsplit=1)
            u = expand_dots({post: v})
            if pre in newd:
                newd[pre] = dict_recursive_update(newd[pre], u)
            else:
                newd[pre] = u
        else:
            newd[k] = v
    return newd


def allbut(mapping, keys):
    mapping = copy.deepcopy(mapping)
    for k in keys:
        del mapping[k]
    return mapping
