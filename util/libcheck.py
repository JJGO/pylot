def check_libjpeg_turbo():
    from PIL import features

    return features.check_feature("libjpeg_turbo")


def check_pillow_simd():
    import PIL

    return ".post" in PIL.__version__


def check_numpy_mkl():
    import numpy as np

    return hasattr(np, "__mkl_version__")


def check_scipy_mkl():
    from .capture import Capturing
    import scipy

    with Capturing() as output:
        scipy.show_config()
    return any("mkl_rt" in line for line in output)


def check_torch_cuda():
    import torch

    return torch.cuda.is_available()


def check_jax_notcpu():
    import jax

    return jax.default_backend() in ["gpu", "tpu"]


def check_environment():
    from warnings import warn

    if not check_numpy_mkl():
        warn("Intel MKL extensions not available for NumPy")
    if not check_scipy_mkl():
        warn("Intel MKL extensions not available for SciPy")
    if not check_libjpeg_turbo():
        warn("libjpeg_turbo not enabled for Pillow")
    if not check_pillow_simd():
        warn("Using slow Pillow instead of Pillow-SIMD")
    if not check_torch_cuda():
        warn("PyTorch cannot find a valid GPU device, check CUDA_VISIBLE_DEVICES")
    if not check_jax_notcpu():
        warn("JAX running on the CPU")
