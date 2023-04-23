import numpy as np
from PIL import Image, ImageChops, ImageOps

import torch


def RGB_project(x):
    C, H, W = x.shape
    # if C % 3 == 0:
    #     x = x.reshape((3, -1, H, W))
    #     return x.sum(axis=1)
    if C == 3:
        return x
    if C == 2:
        y = torch.zeros(3, H, W).to(x.device)
        y[0] = x[0]
        y[1] = x[1]
        return y
    else:
        return x.sum(axis=0, keepdim=True).repeat(3, 1, 1)


def renorm(im, min=0, max=255):
    return min + (im - np.min(im)) / (np.max(im) - np.min(im)) * (max - min)


def torch_renorm(im, min=0, max=255):
    return min + (im - im.min()) / (im.max() - im.min()) * (max - min)


def toImg(im, norm=False):
    im = im.detach().cpu().numpy().transpose((1, 2, 0))
    if im.shape[-1] == 1:
        im = im[..., 0]
    if norm:
        im = renorm(im)
    return Image.fromarray(im.astype(np.uint8))


def trim(im):
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def pad(old_im, new_size):
    old_size = old_im.size
    dw = new_size[0] - old_size[0]
    dh = new_size[1] - old_size[1]
    padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
    new_im = ImageOps.expand(old_im, padding)
    return new_im


def hist_equalize(img: np.ndarray):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")

    return cdf[img]
