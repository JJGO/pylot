# from https://gist.github.com/rwightman/5a7c9232eb57da20afa80cedb8ac2d38
import torch.utils.data as data
import os
import re
import torch
import tarfile
from PIL import Image

from ..util import autoload


IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _extract_tar_info(tarfile, prefix=None):
    class_to_idx = {}
    files = []
    targets = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        if prefix is not None and not dirname.startswith("./" + prefix):
            continue
        target = os.path.basename(dirname)
        class_to_idx[target] = None
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            targets.append(target)
    for idx, c in enumerate(sorted(class_to_idx.keys(), key=natural_key)):
        class_to_idx[c] = idx
    tarinfo_and_targets = zip(files, [class_to_idx[t] for t in targets])
    tarinfo_and_targets = sorted(
        tarinfo_and_targets, key=lambda k: natural_key(k[0].path)
    )
    return tarinfo_and_targets


class ImageFolderTar(data.Dataset):
    def __init__(self, root, transform=None, prefix=None, index=None):

        assert os.path.isfile(root)
        self.root = root
        if index is not None:
            self.imgs = autoload(index)
        else:
            with tarfile.open(
                root
            ) as tf:  # cannot keep this open across processes, reopen later
                self.imgs = _extract_tar_info(tf, prefix=prefix)
        self.tarfile = None  # lazy init in __getitem__
        self.transform = transform
        self.targets = torch.Tensor([i for _, i in self.imgs])

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.imgs[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = Image.open(iob).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
