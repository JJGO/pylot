import copy
import hashlib
import json

def make_hash(o):
    raise ValueError('hash is invalid across runtimes')
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).

    Note that this is not deterministic since Python's hash function isn't deterministic
    """

    if isinstance(o, (set, tuple, list)):

        return tuple([make_hash(e) for e in o])

    elif not isinstance(o, dict):

        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))


def json_digest(data) -> str:
    h = hashlib.sha256(json.dumps(data).encode()).hexdigest()
    return h
