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

def any_getattr(modules, attr):
    for module in reversed(modules):
        if hasattr(module, attr):
            return getattr(module, attr)
    raise AttributeError(f"Attribute {attr} not found in any of {modules}")

