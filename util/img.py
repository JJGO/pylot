import numpy as np
from PIL import Image


def RGB_project(x):
    C, H, W = x.shape
    if C % 3 == 0:
        x = x.reshape((3, -1, H, W))
        return x.sum(axis=1)
    else:
        return x.sum(axis=0, keepdim=True).repeat(3, 1, 1)


def renorm(im, min=0, max=255):
    return min + (im - np.min(im)) / (np.max(im) - np.min(im)) * (max - min)


def torch_renorm(im, min=0, max=255):
    return min + (im - im.min()) / (im.max() - im.min()) * (max - min)


def toImg(im):
    im = im.detach().cpu().numpy().transpose((1, 2, 0))
    if im.shape[-1] == 1:
        im = im[..., 0]
    #     x = renorm(im)
    x = im
    return Image.fromarray(x.astype(np.uint8))
