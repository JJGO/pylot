import random
from typing import Optional, Union, Tuple, Dict

import torch
import numpy as np
import kornia as K
import kornia.augmentation as KA
from kornia.constants import BorderType

from pydantic import validate_arguments

from .common import AugmentationBase2D, _as_single_val, _as2tuple


class RandomBrightnessContrast(AugmentationBase2D):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.0,
        contrast: Union[float, Tuple[float, float]] = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim,
        )
        self.brightness = brightness
        self.contrast = contrast

    def generate_parameters(self, input_shape: torch.Size):

        brightness = _as_single_val(self.brightness)
        contrast = _as_single_val(self.contrast)

        order = np.random.permutation(2)

        return dict(brightness=brightness, contrast=contrast, order=order)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        transforms = [
            lambda img: K.enhance.adjust_brightness(img, params["brightness"]),
            lambda img: K.enhance.adjust_contrast(img, params["contrast"]),
        ]

        jittered = input
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered


class FilterBase(AugmentationBase2D):
    @validate_arguments
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        sigma: Union[float, Tuple[float, float]],
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim,
        )
        self.kernel_size = kernel_size
        self.sigma = sigma


class VariableFilterBase(FilterBase):

    """Helper class for tasks that involve a random filter"""

    def generate_parameters(self, input_shape: torch.Size):
        kernel_size = _as_single_val(self.kernel_size)
        sigma = _as_single_val(self.sigma)
        return dict(kernel_size=kernel_size, sigma=sigma)


class RandomVariableGaussianBlur(VariableFilterBase):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        sigma: Union[float, Tuple[float, float]],
        border_type: str = "reflect",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            sigma=sigma,
            p=p,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
        )
        self.flags = dict(border_type=BorderType.get(border_type))

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        kernel_size = _as2tuple(self.kernel_size)
        sigma = _as2tuple(self.sigma)

        return K.filters.gaussian_blur2d(
            input, kernel_size, sigma, self.flags["border_type"].name.lower()
        )


class RandomVariableBoxBlur(AugmentationBase2D):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        border_type: str = "reflect",
        normalized: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim,
        )
        self.flags = dict(border_type=border_type, normalized=normalized)

    def generate_parameters(self, input_shape: torch.Size):
        kernel_size = _as_single_val(self.kernel_size)
        return Dict(kernel_size=kernel_size)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(params["kernel_size"])
        return K.filters.box_blur(
            input, kernel_size, self.flags["border_type"], self.flags["normalized"]
        )


class RandomVariableGaussianNoise(AugmentationBase2D):
    def __init__(
        self,
        mean: Union[float, Tuple[float, float]] = 0.0,
        std: Union[float, Tuple[float, float]] = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim,
        )
        self.mean = mean
        self.std = std

    def generate_parameters(self, input_shape: torch.Size):

        mean = _as_single_val(self.mean)
        std = _as_single_val(self.std)
 
        if torch.cuda.is_available():
            noise = torch.cuda.FloatTensor(input_shape).normal_(mean, std)
        else:
            noise = torch.FloatTensor(input_shape).normal_(mean, std)

        return dict(noise=noise)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return input + params["noise"]


class RandomVariableElasticTransform(AugmentationBase2D):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 63,
        sigma: Union[float, Tuple[float, float]] = 32,
        alpha: Union[float, Tuple[float, float]] = 1.0,
        align_corners: bool = False,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim,
        )
        self.flags = dict(
            kernel_size=kernel_size,
            sigma=sigma,
            alpha=alpha,
            align_corners=align_corners,
            mode=mode,
            padding_mode=padding_mode,
        )

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        B, _, H, W = shape
        if self.same_on_batch:
            noise = torch.rand(1, 2, H, W, device=self.device, dtype=self.dtype).repeat(
                B, 1, 1, 1
            )
        else:
            noise = torch.rand(B, 2, H, W, device=self.device, dtype=self.dtype)

        kernel_size = _as_single_val(self.flags["kernel_size"])
        sigma = _as_single_val(self.flags["sigma"])
        alpha = _as_single_val(self.flags["alpha"])

        return dict(
            noise=noise * 2 - 1, kernel_size=kernel_size, sigma=sigma, alpha=alpha
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return K.geometry.transform.elastic_transform2d(
            input,
            params["noise"].to(input),
            _as2tuple(params["kernel_size"]),
            _as2tuple(params["sigma"]),
            _as2tuple(params["alpha"]),
            self.flags["align_corners"],
            self.flags["mode"],
            self.flags["padding_mode"],
        )
