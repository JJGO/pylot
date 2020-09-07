import os
import pathlib

WEIGHTS_DIR = "../pretrained"

# TODO make it like dataset path
def weights_path(model, path=None):

    if path is None:
        path = WEIGHTS_DIR
        # Look for the dataset in known paths
        if "WEIGHTSPATH" in os.environ:
            path = os.environ["WEIGHTSPATH"] + ":" + path
    paths = [pathlib.Path(p) for p in path.split(":")]

    for p in paths:
        for root, dirs, files in os.walk(p, followlinks=True):
            if model in files:
                wpath = pathlib.Path(root) / model
                print(f"Found {model} under {wpath}")
                return wpath
    else:
        raise LookupError(f"Could not find {model} in {paths}")


from .head import replace_head, get_classifier_module
from .mnistnet import MnistNet
from .cifar_resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from .cifar_vgg import vgg_bn_drop
from .fresnet import *
