# Some inspiration from:
# - https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py
# - https://junyonglee.me/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
# - https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
# Some background on LMDB: https://blogs.kolabnow.com/2018/06/07/a-short-guide-to-lmdb


import io, math
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable, Any, Literal

import numpy as np
from PIL import Image
import lmdb
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm.auto import tqdm

from pylot.util.ioutil import autoencode, autodecode
from pylot.datasets.cache import IndexedImageFolder


class ImageFolderLMDB(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        env = lmdb.open(str(self.root), readonly=True, lock=False)
        with env.begin(write=False) as txn:
            index = autodecode(txn.get(b"_index.msgpack"), ".msgpack")
            self.__dict__.update(index)

        self.samples = list(zip(self.paths, self.targets))

    @property
    def class_to_idx(self):
        return {k: i for i, k in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if not hasattr(self, "_env"):
            self._env = lmdb.open(
                str(self.root), readonly=True, lock=False, readahead=False, meminit=False
            )
            self._txn = self._env.begin(write=False)

        path = self.paths[i]
        target = self.targets[i]

        # with self._env.begin(write=False) as txn:
        #     _bytes = txn.get(path.encode())
        _bytes = self._txn.get(path.encode())

        if self.write_mode == "compressed":
            sample = Image.open(io.BytesIO(_bytes))
            if sample.mode == 'RGBA':
                sample = sample.convert('RGB')
        else:
            sample = Image.fromarray(autodecode(_bytes, ".msgpack"))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def extra_repr(self) -> str:
        compression = {"compressed": "JPG", "raw": "RAW"}[self.write_mode]
        return f"Compression: {compression}"

    @classmethod
    def fromfolder(
        cls,
        folder: str,
        outfile: str,
        max_workers: int = 16,
        write_mode: Literal["compressed", "raw"] = "compressed",
    ) -> "ImageFolderLMDB":

        dataset = IndexedImageFolder(folder)
        relpaths = dataset.filepaths
        paths, labels = zip(*dataset.samples)

        index = {
            "classes": dataset.classes,
            "paths": relpaths,
            "targets": labels,
            "write_mode": write_mode,
        }

        global _load  # needed to trick concurrent executor

        def _load(path):
            if write_mode == "compressed":
                with open(path, "rb") as f:
                    return f.read()
            else:
                x = np.array(Image.open(path).convert("RGB"))
                return autoencode(x, ".msgpack")

        # We use lmdbm for writing only as it makes autogrowing the DB easier
        from lmdbm import Lmdb

        with Lmdb.open(outfile, "c", map_size=2**32) as db:
            db["_index.msgpack"] = autoencode(index, "_index.msgpack")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, image_bytes in enumerate(
                    tqdm(executor.map(_load, paths), total=len(dataset))
                ):
                    db[relpaths[i]] = image_bytes
        return cls(outfile)


    @classmethod
    def fromdataset(
        cls,
        dataset: VisionDataset,
        outfile: str,
        max_workers: int = 0,
    ) -> 'ImageFolderLMDB':


        classes = dataset.classes if hasattr(dataset, 'classes') else sorted(np.unique(dataset.targets).tolist())

        global _load

        def _load(idx):
            image, _ = dataset[idx]
            array = np.array(image.convert('RGB'))
            return autoencode(array, '.msgpack')

        N = math.ceil(math.log10(len(dataset)))

        # We use lmdbm for writing only as it makes autogrowing the DB easier
        from lmdbm import Lmdb
        with Lmdb.open(outfile, 'c', map_size=2**32) as db:

            paths = []

            if max_workers > 0:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for i, image_bytes in enumerate(
                        tqdm(executor.map(_load, range(len(dataset))), total=len(dataset))
                    ):
                        path = f'{i:0{N}d}'
                        db[path] = image_bytes
                        paths.append(path)
            else:
                for i in tqdm(range(len(dataset))):
                    path = f'{i:0{N}d}'
                    db[path] = _load(i)
                    paths.append(path)


            index = {
                'classes': classes,
                'paths': paths,
                'targets': dataset.targets,
                'write_mode': 'raw',
            }

            db["_index.msgpack"] = autoencode(index, "_index.msgpack")

        return cls(outfile)



if __name__ == "__main__":
    import typer
    from pathlib import Path
    import sys

    def convert(
        source_folder: Path,
        outfile: Path,
        max_workers: int = typer.Option(16, "-P", "--max-workers"),
        write_mode: str = "compressed",
    ):
        if not source_folder.exists():
            print(f"Source path {source_folder} does not exist")
            sys.exit(1)
        outfile = outfile.with_suffix(".lmdb")
        ImageFolderLMDB.fromfolder(
            str(source_folder),
            str(outfile),
            max_workers=max_workers,
            write_mode=write_mode,
        )

    typer.run(convert)
