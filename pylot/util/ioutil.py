import gzip
import io
import json
import pathlib
import pickle
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Literal, Optional, Union

import lz4.frame
import msgpack
import msgpack_numpy as m
import numpy as np
import PIL.Image
import PIL.JpegImagePlugin
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import scipy.io
import xxhash
import yaml
import zstd

import pandas as pd
import torch

m.patch()


class FileExtensionError(Exception):
    pass


class FileFormat(ABC):

    """
    Base class that other formats inherit from
    children formats must overload one of (save|encode) and
    one of (load|decode), the others will get mixin'ed
    """

    EXTENSIONS = []

    @classmethod
    def check_fp(cls, fp):
        if isinstance(fp, io.BytesIO):
            return fp
        if isinstance(fp, str):
            fp = pathlib.Path(fp)
        if fp.suffix not in cls.EXTENSIONS:
            msg = f"{cls.__name__} expects formats {cls.EXTENSIONS}, received {fp.suffix} instead"
            raise FileExtensionError(msg)
        return fp

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("wb") as f:
            f.write(cls.encode(obj))

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("rb") as f:
            return cls.decode(f.read())

    @classmethod
    def encode(cls, obj) -> bytes:
        mem = io.BytesIO()
        cls.save(obj, mem)
        return mem.getvalue()

    @classmethod
    def decode(cls, data: bytes) -> object:
        mem = io.BytesIO(data)
        return cls.load(mem)


class NpyFormat(FileFormat):

    EXTENSIONS = [".npy", ".NPY"]

    @classmethod
    def save(cls, obj: np.ndarray, fp):
        fp = cls.check_fp(fp)
        np.save(fp, obj)

    @classmethod
    def load(cls, fp) -> np.ndarray:
        fp = cls.check_fp(fp)
        return np.load(fp)


class NpzFormat(FileFormat):

    EXTENSIONS = [".npz", ".NPZ"]

    @classmethod
    def save(cls, obj: np.ndarray, fp):
        fp = cls.check_fp(fp)
        np.savez(fp, **obj)

    @classmethod
    def load(cls, fp) -> np.ndarray:
        fp = cls.check_fp(fp)
        return dict(np.load(fp))


class PtFormat(FileFormat):

    EXTENSIONS = [".pt", ".PT", ".pth", ".PTH"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        torch.save(obj, fp)

    @classmethod
    def load(cls, fp):
        fp = cls.check_fp(fp)
        return torch.load(fp)


class YamlFormat(FileFormat):

    EXTENSIONS = [".yaml", ".YAML", ".yml", ".YML"]

    @classmethod
    def encode(cls, obj) -> str:
        return yaml.safe_dump(obj, indent=2).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> object:
        return yaml.safe_load(data)

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("w") as f:
            yaml.safe_dump(obj, f)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("r") as f:
            return yaml.safe_load(f)


class NumpyJSONEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonFormat(FileFormat):

    EXTENSIONS = [".json", ".JSON"]

    @classmethod
    def encode(cls, obj) -> bytes:
        return json.dumps(obj, cls=NumpyJSONEncoder).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> object:
        return json.loads(data.decode("utf-8"))

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("w") as f:
            json.dump(obj, f, cls=NumpyJSONEncoder)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("r") as f:
            return json.load(f)


class JsonlFormat(FileFormat):

    EXTENSIONS = [".jsonl", ".JSONL"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        if isinstance(obj, list):
            with fp.open('w') as f:
                print(
                    "\n".join((json.dumps(row) for row in obj)),
                    file=f
                )
            return
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Can only serialize pd.DataFrame objects")
        fp = cls.check_fp(fp)
        obj.to_json(fp, orient="records", lines=True)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        return pd.read_json(fp, lines=True)


class CsvFormat(FileFormat):

    EXTENSIONS = [".csv", ".CSV"]

    @classmethod
    def save(cls, obj, fp):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Can only serialize pd.DataFrame objects")
        fp = cls.check_fp(fp)
        obj.to_csv(fp, index=False)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        return pd.read_csv(fp)


# class ParquetFormat(FileFormat):

#     EXTENSIONS = [".parquet", ".PARQUET", ".pq", ".PQ"]

#     @classmethod
#     def save(cls, obj, fp):
#         if not isinstance(obj, pd.DataFrame):
#             raise TypeError("Can only serialize pd.DataFrame objects")
#         fp = cls.check_fp(fp)
#         obj.to_parquet(fp, index=False)

#     @classmethod
#     def load(cls, fp) -> object:
#         return pd.read_parquet(fp)
#         fp = cls.check_fp(fp)


class ParquetFormat(FileFormat):

    # Tweaked version of write/read_parquet to support
    # storing .attrs metadata in the parquet metadata fields

    EXTENSIONS = [".parquet", ".PARQUET", ".pq", ".PQ"]

    @classmethod
    def save(cls, obj, fp):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Can only serialize pd.DataFrame objects")
        fp = cls.check_fp(fp)
        table = pa.Table.from_pandas(obj)
        meta = json.dumps(obj.attrs)
        new_meta = {b"custom_metadata": meta, **table.schema.metadata}
        table = table.replace_schema_metadata(new_meta)
        pq.write_table(table, fp)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        table = pq.read_table(fp)
        df = table.to_pandas()
        custom_meta = table.schema.metadata.get(b"custom_metadata", "{}")
        df.attrs = json.loads(custom_meta)
        return df


class FeatherFormat(FileFormat):
    # TODO feather supports lz4 and zstd natively, reject wrapper-compression
    EXTENSIONS = [".feather", ".FEATHER"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        feather.write_feather(obj, fp)

    @classmethod
    def load(cls, fp) -> object:
        return feather.read_feather(fp)


class PickleFormat(FileFormat):

    EXTENSIONS = [".pkl", ".PKL", ".pickle", ".PICKLE"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        with fp.open("wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def encode(cls, obj) -> bytes:
        return pickle.dumps(obj)

    @classmethod
    def decode(cls, data: bytes) -> object:
        return pickle.loads(data)


class BaseImageFormat(FileFormat):
    @classmethod
    def encode(cls, obj: PIL.Image):
        if (
            isinstance(obj, PIL.Image.Image)
            and hasattr(obj, "filename")
            and ("." + obj.format) in cls.EXTENSIONS
        ):
            # avoid re-encoding, relevant for JPEGs
            orig = pathlib.Path(obj.filename)
            with orig.open("rb") as src:
                data = src.read()
            return data

        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().detach()
        if isinstance(obj, np.ndarray):
            obj = PIL.Image.fromarray(obj)
        if not isinstance(obj, PIL.Image.Image):
            raise TypeError(
                "Can only serialize PIL.Image|np.ndarray|torch.Tensor objects"
            )
        mem = io.BytesIO()
        obj.save(mem, format=cls.FORMAT)
        return mem.getvalue()

    @classmethod
    def load(cls, fp) -> PIL.Image:
        fp = cls.check_fp(fp)
        return PIL.Image.open(fp)


class JPGFormat(BaseImageFormat):

    EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG"]
    FORMAT = "jpeg"


class PNGFormat(BaseImageFormat):

    EXTENSIONS = [".png", ".PNG"]
    FORMAT = "png"


class GzipFormat(FileFormat):

    EXTENSIONS = [".gz", ".GZ"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return gzip.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return gzip.decompress(data)


class LZ4Format(FileFormat):

    EXTENSIONS = [".lz4", ".LZ4"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return lz4.frame.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return lz4.frame.decompress(data)


class ZstdFormat(FileFormat):

    EXTENSIONS = [".zst", ".ZST"]

    @classmethod
    def encode(cls, data: bytes) -> bytes:
        return zstd.compress(data)

    @classmethod
    def decode(cls, data: bytes) -> bytes:
        return zstd.decompress(data)


class MsgpackFormat(FileFormat):

    EXTENSIONS = [".msgpack", ".MSGPACK"]

    @classmethod
    def encode(cls, data: Any) -> bytes:
        return msgpack.packb(data)

    @classmethod
    def decode(cls, data: bytes) -> Any:
        return msgpack.unpackb(data)


class PlaintextFormat(FileFormat):

    EXTENSIONS = [".txt", ".TXT", ".log", ".LOG"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        if isinstance(obj, list):
            obj = "\n".join(obj)
        with fp.open("w") as f:
            f.write(obj)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        with fp.open("r") as f:
            return f.read().strip().split("\n")


class MatFormat(FileFormat):

    EXTENSIONS = [".mat", ".MAT"]

    @classmethod
    def save(cls, obj, fp):
        fp = cls.check_fp(fp)
        scipy.io.savemat(fp, obj)

    @classmethod
    def load(cls, fp) -> object:
        fp = cls.check_fp(fp)
        return scipy.io.loadmat(fp)


_DEFAULTFORMAT = {}
for format_cls in (
    NpyFormat,
    NpzFormat,
    PtFormat,
    YamlFormat,
    JsonFormat,
    JsonlFormat,
    CsvFormat,
    ParquetFormat,
    FeatherFormat,
    PickleFormat,
    PNGFormat,
    JPGFormat,
    GzipFormat,
    LZ4Format,
    ZstdFormat,
    MsgpackFormat,
    PlaintextFormat,
    MatFormat,
):
    for extension in format_cls.EXTENSIONS:
        extension = extension.strip(".")
        assert extension not in _DEFAULTFORMAT
        _DEFAULTFORMAT[extension] = format_cls


class InvalidExtensionError(Exception):

    valid_extensions = "|".join(list(_DEFAULTFORMAT))

    def __init__(self, extension):
        message = f"Unsupported extension '.{extension}'"
        super().__init__(message)
        self.extension = extension


def autoencode(obj: object, filename: str) -> bytes:
    _, *exts = str(filename).split(".")
    data = obj
    for ext in exts:
        if ext not in _DEFAULTFORMAT:
            raise InvalidExtensionError(ext)
        data = _DEFAULTFORMAT[ext].encode(data)
    return data


def autodecode(data: bytes, filename: str) -> object:
    _, *exts = str(filename).split(".")
    for ext in reversed(exts):
        if ext not in _DEFAULTFORMAT:
            raise InvalidExtensionError(ext)
        data = _DEFAULTFORMAT[ext].decode(data)
    return data


def autoload(path: Union[str, pathlib.Path]) -> object:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file {str(path)}")

    ext = path.suffix.strip(".")
    if ext not in _DEFAULTFORMAT:
        raise InvalidExtensionError(ext)
    if ext.lower() in ("lz4", "zst", "gz"):
        return autodecode(_DEFAULTFORMAT[ext].load(path), path.stem)
    return _DEFAULTFORMAT[ext].load(path)


def autosave(obj, path: Union[str, pathlib.Path], parents=True) -> object:
    if isinstance(path, str):
        path = pathlib.Path(path)
    ext = path.suffix.strip(".")
    if ext not in _DEFAULTFORMAT:
        raise InvalidExtensionError(ext)
    if parents:
        path.parent.mkdir(exist_ok=True, parents=True)
    if ext.lower() in ("lz4", "zst", "gz"):
        obj = autoencode(obj, path.stem)
    return _DEFAULTFORMAT[ext].save(obj, path)


_ENCODERS = {
    np.ndarray: ".msgpack.lz4",
    torch.Tensor: ".pt.lz4",
    PIL.JpegImagePlugin.JpegImageFile: ".jpg",
    PIL.Image.Image: ".png",
    pd.DataFrame: ".feather",
    (str, list, tuple, dict, int, float, bool, bytes): ".msgpack.lz4",
    object: ".pkl.lz4",
}


def autopackb(obj: object) -> bytes:
    for types, ext in _ENCODERS.items():
        if isinstance(obj, types):
            break
    data = autoencode(obj, ext)
    # doing ext || null || payload is faster than msgpack-ing it
    return ext.encode() + b"\x00" + data


def autounpackb(data: bytes) -> object:
    ext, _, data = data.partition(b"\x00")
    return autodecode(data, ext.decode())


def autohash(value: object) -> str:
    data = autopackb(value)
    return xxhash.xxh3_64_hexdigest(data)


@contextmanager
def inplace_edit(file, backup=False):
    if isinstance(file, str):
        file = pathlib.Path(file)
    obj = autoload(file)
    yield obj
    if backup:
        shutil.copy(file, file.with_suffix(file.suffix + ".bk"))
    autosave(obj, file)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
