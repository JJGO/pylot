import os
import pathlib
import tempfile
import shutil
import uuid
from contextlib import contextmanager
from typing import Optional

import s3fs

_s3driver = None


def make_path(path):
    if path.startswith("s3://"):
        S3Path.init_driver(
            os.environ["S3_HOST"], os.environ["S3_KEY"], os.environ["S3_SECRET"]
        )
        return S3Path(path[len("s3://") :])
    return pathlib.Path(path)


class S3Path(pathlib.PosixPath):

    _s3driver: Optional[s3fs.S3FileSystem] = None

    @classmethod
    def get_driver(cls):
        assert (
            cls._s3driver is not None
        ), "Driver must be initialized with S3Path.init_driver"
        return cls._s3driver

    @classmethod
    def init_driver(cls, host, key, secret):
        if cls._s3driver is None:
            cls._s3driver = s3fs.S3FileSystem(
                anon=False,
                client_kwargs=dict(endpoint_url=host),
                key=key,
                secret=secret,
            )

    # def __new__(cls, *args, **kwargs):
    def _init(self, *args, **kwargs):
        if self._s3driver is None and all(
            k in os.environ for k in ("S3_HOST", "S3_KEY", "S3_SECRET")
        ):
            S3Path.init_driver(
                os.environ["S3_HOST"], os.environ["S3_KEY"], os.environ["S3_SECRET"]
            )
        assert (
            self._s3driver is not None
        ), "Driver must be initialized with S3Path.init_driver"
        super()._init(*args, **kwargs)

    def chmod(self, mode):
        return self._s3driver.chmod(str(self), mode)

    def exists(self):
        return self._s3driver.exists(str(self))

    def glob(self, pattern):
        for p in self._s3driver.glob(str(self) + "/" + pattern):
            yield p

    def mkdir(self, parents=False, exist_ok=True):
        if self.exists() and not exist_ok:
            raise ValueError("Folder already exists")
        if not self.exists():
            (self / "._s3").touch()
        # self._s3driver.mkdir(str(self), create_parents=parents)

    @contextmanager
    def open(self, mode="r"):
        with self._s3driver.open(str(self), mode) as f:
            yield f

    def rename(self, target):
        if isinstance(target, str):
            target = S3Path(target)
        assert isinstance(target, S3Path)
        self._s3driver.rename(str(self), str(target))
        return target

    def rmdir(self):
        (self / "._s3").unlink(missing_ok=True)
        # self._s3driver.rmdir(str(self))

    def stat(self):
        return self._s3driver.stat(str(self))

    def touch(self):
        self._s3driver.touch(str(self), truncate=False)

    def iterdir(self):
        for path in self._s3driver.ls(str(self)):
            yield S3Path(path)

    def unlink(self, missing_ok=False):
        try:
            self._s3driver.rm(str(self))
        except FileNotFoundError:
            if missing_ok:
                return
            raise

    def is_file(self):
        return self._s3driver.is_file(str(self))

    def is_dir(self):
        return self._s3driver.is_dir(str(self))

    def rmtree(self):
        return self._s3driver.rm(str(self), recursive=True)

    @staticmethod
    @contextmanager
    def as_local(remote_path):
        if isinstance(remote_path, S3Path):
            local_path = pathlib.Path("/tmp/" + uuid.uuid4().hex)
            local_path = local_path.with_suffix(remote_path.suffix)

            if remote_path.exists():
                remote_path._s3driver.get(
                    str(remote_path), str(local_path), recursive=True
                )

            yield local_path
            remote_path._s3driver.put(str(local_path), str(remote_path), recursive=True)

            if local_path.is_file():
                local_path.unlink()
            else:
                shutil.rmtree(local_path)
        else:
            yield remote_path

    def __repr__(self):
        return f"{self.__class__.__name__}('{super().__str__()}')"

    def __str__(self):
        return f"s3://{super().__str__()}"


# @contextmanager
# def local_path(path):
#     if isinstance(path, S3Path):
#         local_path = pathlib.Path('/tmp/' + uuid.uuid4().hex)
#         local_path = local_path.with_suffix(path.suffix)
#         yield local_path
#         driver = path.get_driver()
#         driver.put(str(local_path), str(path), recursive=True)
#         if local_path.is_file():
#             local_path.unlink()
#         else:
#             shutil.rmtree(local_path)
#     else:
#         yield path
