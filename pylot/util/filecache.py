import pathlib

from concurrent.futures import ThreadPoolExecutor
from diskcache import Cache, Disk

from .ioutil import autoload


class FileCache(Cache):
    def __init__(self, directory=None, timeout=60, disk=Disk, **settings):
        super().__init__(directory=directory, timeout=timeout, disk=disk, **settings)

    def get(self, file, **kwargs):
        if isinstance(file, str):
            file = pathlib.Path(file)
        file = file.absolute()
        if not file.exists():
            return None
            # raise FileNotFoundError(f"No such file {str(file)}")
        last_modified = file.stat().st_mtime
        last_saved = super().get((file, "mtime"), None)
        if file not in self or last_modified != last_saved:
            super().set(file, autoload(file))
            super().set((file, "mtime"), last_modified)
        return super().get(file, **kwargs)

    def gets(self, files, num_workers=0, default=None):
        if num_workers == 0:
            return [self.get(file) for file in files]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(self.get, files))
