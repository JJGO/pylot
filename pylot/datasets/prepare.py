# From albumentations example
# https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
from urllib.request import urlretrieve
import os
import pathlib
import shutil
from tqdm.auto import tqdm

# torchvision datasets provides similar (and more sophisticated) tools under torchvision.datasets.utils


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


def download_and_extract_archive(url, root):
    filename = os.path.basename(url)
    filename = os.path.join(root, filename)
    print(f"Downloading {url} to {filename}")
    download_url(url, filename)
    print(f"Extracting {filename} to {root}")
    extract_archive(filename)
