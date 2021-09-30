import pathlib

import numpy as np
from torch import Tensor
import zarr
import zarr.hierarchy
import zarr.core


class TensorStore:
    def __init__(self, path):
        self.datapath = pathlib.Path(path)
        self.root = zarr.open(str(self.datapath), mode="a")

    def put(self, data, *args, force=True):
        path = "/".join(map(str, args))
        if path in self.root:
            del self.root[path]
        self[path] = data

    def __setitem__(self, path, data, force=False):

        if force and path in self.root:
            del self.root[path]

        if isinstance(data, Tensor):
            data = data.detach().cpu().numpy()

        if isinstance(data, np.ndarray):
            self.root.create_dataset(path, data=data)
            return

        # TODO: add support for dataframes

        if isinstance(data, tuple):
            data = list(data)

        if isinstance(data, list):
            for i, x in enumerate(data):
                self.put(x, path, i)
            self.root["path"].attrs["type"] = "list"
            return

        if isinstance(data, dict):
            for k, v in data.items():
                self.put(v, path, k)
            return

        raise ValueError(f"Cannot save type {type(data)}")

    def get(self, *args):
        path = "/".join(map(str, args))
        return self[path]

    def __getitem__(self, path):

        if path != "":
            x = self.root[path]
        else:
            x = self.root

        type_ = x.attrs.get("type", None)

        if type_ == "list":
            return [self.get(path, i) for i in range(len(self.root))]

        if isinstance(x, zarr.hierarchy.Group):
            return {k: self.get(path, k) for k in x}
        if isinstance(x, zarr.core.Array):
            return x[:]

    def tree(self):
        # self.root.visit(print)
        return self.root.tree()
