import os
from typing import Optional, Callable

import torch
import numpy as np
from torchvision.datasets import MNIST, USPS
from tqdm.auto import tqdm


class USPS28(MNIST):
    def _load_data(self):
        split = "train" if self.train else "test"

        file = os.path.join(self.root, f"USPS28/usps28-{split}.npz")
        data = np.load(file)
        imgs = torch.from_numpy(data["imgs"])
        targets = torch.from_numpy(data["targets"])
        return imgs, targets

    def download(self):
        import PIL

        dst_folder = os.path.join(self.root, self.__class__.__name__)
        os.makedirs(dst_folder, exist_ok=True)

        for split in ["train", "test"]:
            resized_images = []
            d = USPS(self.root / "USPS", train=(split == "train"))
            for i in tqdm(d):
                r = i[0].resize((28, 28), PIL.Image.NEAREST)
                resized_images.append(np.array(r))
            resized_images = np.array(resized_images)
            file = os.path.join(dst_folder, f"usps28-{split}.npz")
            targets = np.array(d.targets)
            np.savez_compressed(file, imgs=resized_images, targets=targets)

    def _check_exists(self):
        import pathlib

        return all(
            (pathlib.Path(self.root) / "USPS28" / f"usps28-{split}.npz").exists()
            for split in ["train", "test"]
        )
