import os
import pathlib


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- Name of the dataset

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
                f"No path specified. A path must be provided, or the folder must be listed in your DATAPATH"
            )

    paths = [pathlib.Path(p) for p in path.split(":")]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            return p
    raise LookupError(f"Could not find {dataset} in {paths}")


class DatapathMixin:
    @property
    def path(self):
        root = getattr(self, "root", None)
        if root:
            return root
        return dataset_path(self._folder_name, root)

    @property
    def _folder_name(self):
        return self.__class__.__name__
