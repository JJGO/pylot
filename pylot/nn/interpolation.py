import torch.nn.functional as F


def resize(x, scale_factor=None, size=None, interpolation_mode="linear"):

    assert (scale_factor is not None) ^ (
        size is not None
    ), "Either scale_factor or size must be specified, not both"

    align_corners = {
        "linear": False,
        "bilinear": False,
        "trilinear": False,
        "nearest": None,
    }[interpolation_mode]

    dims = len(x.shape) - 2  # Batch and channel dim
    mode = ["linear", "bilinear", "trilinear"][dims - 1]

    if size is None:
        size = tuple([max(1, int(dim * scale_factor)) for dim in x.shape[2:]])
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning)
    return F.interpolate(
        x,
        size=size,
        align_corners=align_corners,
        mode=mode,
    )
