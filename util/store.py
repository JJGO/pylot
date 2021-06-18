import atexit
import pathlib

import h5py
import numpy as np
from torch import Tensor
from PIL import Image


class TensorStore:
    def __init__(self, path):
        self.datapath = pathlib.Path(path)
        self.hf = h5py.File(self.datapath, "a")
        atexit(self.hf.close)

    def put(self, data, *args, force=True):
        path = "/".join(map(str, args))
        if path in self.hf:
            del self.hf[path]
        self[path] = data

    def __setitem__(self, path, data):

        if isinstance(data, Tensor):
            data = data.detach().cpu().numpy()

        if isinstance(data, np.ndarray):
            self.hf.create_dataset(path, data=data)
            return

        if isinstance(data, Image.Image):
            self.hf.create_dataset(path, data=np.array(data))
            self.hf[path].attrs["type"] = "Image"
            return

        # TODO: add support for dataframes

        if isinstance(data, tuple):
            data = list(data)

        if isinstance(data, list):
            for i, x in enumerate(data):
                self.put(x, path, i)
            self.hf["path"].attrs["type"] = "list"
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
            x = self.hf[path]
        else:
            x = self.hf

        type_ = x.attrs.get("type", None)

        if isinstance(x, h5py.Dataset):
            if type_ == "Image":
                return Image.fromarray(x[()])
            return x[()]

        if type_ == "list":
            return [self.get(path, i) for i in range(len(self.hf))]

        return {k: self.get(path, k) for k in x}

    def __del__(self):
        if hasattr(self, "hf"):
            self.hf.close()

    def tree(self):
        self.hf.visit(print)
