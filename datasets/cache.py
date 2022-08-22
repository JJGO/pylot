import json
import os
import pathlib

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset, IMG_EXTENSIONS, default_loader


def removeprefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


class IndexedDatasetFolder(VisionDataset):
    def __init__(
        self,
        root,
        loader,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        index = pathlib.Path(root).with_suffix(".json")
        root_prefix = str(root) + "/"
        if not index.exists():
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            if not index.exists():
                # Navigating the FS can take long, some processs might have
                # Finished before we do
                labels = [label for _, label in samples]
                relative_paths = [
                    removeprefix(path, root_prefix) for path, _ in samples
                ]
                cache = {
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "labels": labels,
                    "relative_paths": relative_paths,
                }
                with open(index, "w") as f:
                    json.dump(cache, f)
        else:
            with open(index, "r") as f:
                cache = json.load(f)
                classes, class_to_idx = cache["classes"], cache["class_to_idx"]
                paths = [root_prefix + rel_path for rel_path in cache["relative_paths"]]
                samples = [(path, label) for path, label in zip(paths, cache["labels"])]

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    @property
    def filepaths(self):
        index = pathlib.Path(self.root).with_suffix(".json")
        with index.open("r") as f:
            cache = json.load(f)
        return cache["relative_paths"]


class IndexedImageFolder(IndexedDatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


def IndexedImageDataset(root, train=True, **kwargs):
    root = pathlib.Path(root)
    root /= "train" if train else "val"
    return IndexedImageFolder(root, **kwargs)
