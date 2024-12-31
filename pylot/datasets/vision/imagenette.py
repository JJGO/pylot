from typing import Literal

from ..lmdb import ImageFolderLMDB
from ..path import DatapathMixin

from pydantic import validate_arguments
from ...util.meta import store_attr

__all__ = ["Imagenette", "Imagewoof"]


class ImagenetteBase(ImageFolderLMDB, DatapathMixin):
    def __init__(
        self,
        split: Literal["train", "val"],
        resolution: Literal[320, 160, "full"] = 320,
    ):
        store_attr()
        super().__init__(self.path)  # , transform=self.preproc_transform)


class Imagenette(ImagenetteBase):
    @property
    def _folder_name(self):
        res = {
            "full": "",
            320: "-320",
            160: "-160",
        }[self.resolution]
        return f"Imagenette-lmdb/imagenette2{res}/{self.split}"


class Imagewoof(ImagenetteBase):
    @property
    def _folder_name(self):
        res = {
            "full": "",
            320: "-320",
            160: "-160",
        }[self.resolution]
        return f"Imagenette-lmdb/imagewoof2{res}/{self.split}"


# https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagewang.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagewang-320.tgz
# https://s3.amazonaws.com/fast-ai-imageclas/imagewang-160.tgz
