import pathlib
import os

from torchvision import transforms, datasets
import PIL

from ..indexed import IndexedImageDataset

__all__ = [
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
    "Places365",
    "Miniplaces",
    "TinyImageNet",
]

_constructors = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    "ImageNet": IndexedImageDataset,
    "Places365": IndexedImageDataset,
    "Miniplaces": IndexedImageDataset,
    "TinyImageNet": IndexedImageDataset,
}


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        dataset_path -- pathlib.Path for the first match

    Raises:
        ValueError -- If no path is provided and DATAPATH is not set
        LookupError -- If the given dataset cannot be found
    """
    if path is None:
        # Look for the dataset in known paths
        if "DATAPATH" in os.environ:
            path = os.environ["DATAPATH"]
            paths = [pathlib.Path(p) for p in path.split(":")]
        else:
            raise ValueError(
                f"No path specified. A path must be provided, \n \
                           or the folder must be listed in your DATAPATH"
            )

    paths = [pathlib.Path(p) for p in path.split(":")]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            # print(f"Found {dataset} under {p}")
            return p
    raise LookupError(f"Could not find {dataset} in {paths}")


def dataset_builder(dataset, train=True, normalize=None, preproc=None, path=None):
    """Build a torch.utils.Dataset with proper preprocessing

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        train {bool} -- Whether to return train or validation set (default: {True})
        normalize {torchvision.Transform} -- Transform to normalize data channel wise (default: {None})
        preproc {list(torchvision.Transform)} -- List of preprocessing operations (default: {None})
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        torch.utils.data.Dataset -- Dataset object with transforms and normalization
    """
    if preproc is not None:
        preproc += [transforms.ToTensor()]
        if normalize is not None:
            preproc += [normalize]
        preproc = transforms.Compose(preproc)

    kwargs = {"transform": preproc}

    path = dataset_path(dataset, path)

    return _constructors[dataset](path, train=train, **kwargs)


def MNIST(train=True, path=None, norm=False, augmentation=False, augment_kw=None):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    augment_kwargs = dict(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.9, 1.0),
        shear=(5, 5),
        interpolation=PIL.Image.BILINEAR,
    )
    if augment_kw:
        augment_kwargs.update(augment_kw)
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,)) if norm else None
    preproc = [] if not augmentation else [transforms.RandomAffine(**augment_kwargs)]
    dataset = dataset_builder("MNIST", train, normalize, preproc, path)
    dataset.shape = (1, 28, 28)
    dataset.n_classes = 10
    return dataset


def CIFAR10(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR10"""
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder("CIFAR10", train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    dataset.n_classes = 10
    return dataset


def CIFAR100(train=True, path=None):
    """Thin wrapper around torchvision.datasets.CIFAR100"""
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder("CIFAR100", train, normalize, preproc, path)
    dataset.shape = (3, 32, 32)
    dataset.n_classes = 100
    return dataset


def ImageNet(train=True, path=None):
    """Thin wrapper around IndexedImageDataset"""
    # TODO Better data augmentation?
    # ImageNet loading from files can produce benign EXIF errors
    import warnings

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder("ImageNet", train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    dataset.n_classes = 1000
    return dataset


def Places365(train=True, path=None):
    """Thin wrapper around IndexedImageDataset"""
    # Note : Bolei used the normalization for Imagenet, not the one for Places!
    # # https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py
    # So these are kept so weights are compatible
    # TODO Better data augmentation
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder("Places365", train, normalize, preproc, path)
    dataset.shape = (3, 224, 224)
    dataset.n_classes = 365
    return dataset


def TinyImageNet(train=True, path=None):
    """Thin wrapper around IndexedImageDataset"""
    # TODO Better data augmentation

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomCrop(64, 8), transforms.RandomHorizontalFlip()]
    else:
        preproc = []
    dataset = dataset_builder("TinyImageNet", train, normalize, preproc, path)
    dataset.shape = (3, 64, 64)
    dataset.n_classes = 200
    return dataset


def Miniplaces(train=True, path=None):
    """Thin wrapper around IndexedImageDataset"""
    # TODO compute normalization constants for Miniplaces
    # TODO Better data augmentation

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        preproc = [transforms.RandomCrop(128, 16), transforms.RandomHorizontalFlip()]
    else:
        preproc = []
    dataset = dataset_builder("Miniplaces", train, normalize, preproc, path)
    dataset.shape = (3, 128, 128)
    dataset.n_classes = 100
    return dataset


def nanoImageNet(train=True, path=None):
    from .subset import subset_dataset

    d = ImageNet(train=train, path=path)
    return subset_dataset(d, 10, 1000)
