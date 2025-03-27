from dataclasses import dataclass
from typing import Literal

import torchvision.transforms as TF

from ..indexed import IndexedImageFolder
from ..path import DatapathMixin

from ...util.validation import validate_arguments_init
from ...util import autoload


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIMPLE_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
DETAILED_LABELS_URL = 'https://gist.githubusercontent.com/JJGO/d61c1d1eb161dba58e6354fddca1e99a/raw/3b4962e7fb1213be9627b53c2ce1f4043209efc5/imagenet-detailed-labels.json'


@validate_arguments_init
@dataclass(repr=False)
class ImageNet(IndexedImageFolder, DatapathMixin):

    split: Literal["train", "val"] = "train"
    transforms: bool = True

    def __post_init__(self):
        self._filter_warnings()
        super().__init__(self.path / self.split, transform=self.get_transform())

    def get_transform(self):
        if not self.transforms:
            return None
        if self.split == "train":
            return TF.Compose(
                [
                    TF.RandomResizedCrop(224),
                    TF.RandomHorizontalFlip(),
                    TF.ToTensor(),
                    TF.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        elif self.split == "val":
            return TF.Compose(
                [
                    TF.Resize(256),
                    TF.CenterCrop(224),
                    TF.ToTensor(),
                    TF.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    @staticmethod
    def _filter_warnings():
        # ImageNet loading from files can produce benign EXIF errors
        import warnings

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    @property
    def simple_labels(self):
        file = self.path / "imagenet-simple-labels.json"
        if not file.exists():
            from urllib.request import urlretrieve

            urlretrieve(SIMPLE_LABELS_URL, str(file))
        return autoload(file)

    @property
    def detailed_labels(self):
        file = self.path / "imagenet-detailed-labels.json"
        if not file.exists():
            from urllib.request import urlretrieve

            urlretrieve(DETAILED_LABELS_URL, str(file))
        return autoload(file)
