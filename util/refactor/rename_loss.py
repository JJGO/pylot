import pathlib
from pylot.util import inplace_yaml

def rename_loss(path):
    cpath = pathlib.Path(path) / "config.yml"
    if cpath.exists():
        with inplace_yaml(cpath, backup=True) as cfg:
            if "loss" in cfg["loss"]:
                cfg["loss"]["loss_func"] = cfg["loss"].pop("loss")
                return True
    return False
