import os
from pathlib import Path
from typing import Union
import shutil

from pandas import Series

from .register import register_series_accessor


@register_series_accessor("unix")
class UnixAccessor:
    def __init__(self, ser: Series):
        self._validate(ser)
        self._ser = ser.map(Path)

    @staticmethod
    def _validate(ser):
        assert ser.map(lambda x: isinstance(x, (Path, str))).all()

    @staticmethod
    def _validate_dest(dest: Union[Path, str]) -> Path:
        assert isinstance(dest, (str, Path))
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def rm(self, strict=False):
        ser = self._ser
        if not strict:
            ser = ser[ser.map(lambda x: x.exists())]

        def _rm(path: Path):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        ser.map(_rm)

    def cp(self, dest: Union[str, Path], strict: bool = False):
        dest = self._validate_dest(dest)
        ser = self._ser
        if not strict:
            ser = ser[ser.map(lambda x: x.exists())]

        def _cp(path: Path):
            if path.is_dir():
                shutil.copytree(path, dest / path.name)
            else:
                shutil.copy(path, dest / path.name)

        # ser.map(lambda path: shutil.copy(path, dest / path.name))
        ser.map(_cp)
        return ser.map(lambda path: dest / path.name)

    def mv(self, dest: Union[str, Path], strict: bool = False):
        dest = self._validate_dest(dest)
        ser = self._ser
        if not strict:
            ser = ser[ser.map(lambda x: x.exists())]
        ser.map(lambda path: shutil.move(path, dest / path.name))
        return ser.map(lambda path: dest / path.name)

    def ln(self, dest: Union[str, Path], strict: bool = False):
        dest = self._validate_dest(dest)
        ser = self._ser
        if not strict:
            ser = ser[ser.map(lambda x: x.exists())]
        ser.map(lambda path: os.symlink(path, dest / path.name))
        return ser.map(lambda path: dest / path.name)
