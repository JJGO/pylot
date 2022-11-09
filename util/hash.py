import copy
import hashlib
import json
from pydantic import validate_arguments
import pathlib
import zlib
import xxhash

DEFAULT_CHUNKSIZE = 64 * 1024


def make_hash(o):
    raise ValueError("hash is invalid across runtimes")
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


@validate_arguments
def file_crc(path: pathlib.Path, chunksize=DEFAULT_CHUNKSIZE) -> str:
    crc = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(chunksize)
            if not data:
                break
            crc = zlib.adler32(data, crc)
    crc &= 0xFFFFFFFF
    return f"{crc:x}"


@validate_arguments
def file_digest(path: pathlib.Path, chunksize=DEFAULT_CHUNKSIZE) -> str:
    # Use xxhash as it's substantially faster than md5
    # See https://github.com/Cyan4973/xxHash
    h = xxhash.xxh3_64()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunksize)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


@validate_arguments
def fast_file_digest(path: pathlib.Path) -> str:
    import imohash

    return imohash.hashfile(path, hexdigest=True)
