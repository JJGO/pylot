"""Module for computing FLOPs from specification, not from Torch objects
"""
import numpy as np


def dense_flops(in_neurons, out_neurons):
    """Compute the number of multiply-adds used by a Dense (Linear) layer"""
    return in_neurons * out_neurons


def convNd_flops(
    in_channels,
    out_channels,
    input_shape,
    kernel_shape,
    padding="same",
    strides=1,
    dilation=1,
):
    """Compute the number of multiply-adds used by a Conv2D layer
    Args:
        in_channels (int): The number of channels in the layer's input
        out_channels (int): The number of channels in the layer's output
        input_shape (int, int): The spatial shape of the rank-3 input tensor
        kernel_shape (int, int): The spatial shape of the rank-4 kernel
        padding ({'same', 'valid'}): The padding used by the convolution
        strides (int) or (int, int): The spatial stride of the convolution;
            two numbers may be specified if it's different for the x and y axes
        dilation (int): Must be 1 for now.
    Returns:
        int: The number of multiply-adds a direct convolution would require
        (i.e., no FFT, no Winograd, etc)
    >>> c_in, c_out = 10, 10
    >>> in_shape = (4, 5)
    >>> filt_shape = (3, 2)
    >>> # valid padding
    >>> ret = convNd_flops(c_in, c_out, in_shape, filt_shape, padding='valid')
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 4))
    True
    >>> # same padding, no stride
    >>> ret = convNd_flops(c_in, c_out, in_shape, filt_shape, padding='same')
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * np.prod(in_shape))
    True
    >>> # valid padding, stride > 1
    >>> ret = convNd_flops(c_in, c_out, in_shape, filt_shape, \
                       padding='valid', strides=(1, 2))
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 2))
    True
    >>> # same padding, stride > 1
    >>> ret = convNd_flops(c_in, c_out, in_shape, filt_shape, \
                           padding='same', strides=2)
    >>> ret == int(c_in * c_out * np.prod(filt_shape) * (2 * 3))
    True
    """
    # validate + sanitize input
    assert in_channels > 0
    assert out_channels > 0
    assert len(input_shape) == len(kernel_shape)
    assert len(kernel_shape) in (1, 2, 3)
    padding = padding.lower()
    assert padding in (
        "same",
        "valid",
        "zeros",
    ), "Padding must be one of same|valid|zeros"

    if isinstance(strides, int):
        strides = tuple([strides]*len(input_shape))
    assert len(strides) == len(kernel_shape)

    assert dilation == 1 or all(
        d == 1 for d in dilation
    ), "Dilation > 1 is not supported"

    # compute output spatial shape
    # based on TF computations https://stackoverflow.com/a/37674568
    if padding in ["same", "zeros"]:
        output_shape = tuple([np.ceil(float(i) / s) for i, s in zip(input_shape, strides)])
    else:  # padding == 'valid'
        output_shape = tuple([np.ceil((i - k + 1 ) / s) for i, s, k in zip(input_shape, strides, kernel_shape)]) # noqa

    # work to compute one output spatial position
    nflops = in_channels * out_channels * int(np.prod(kernel_shape))

    # total work = work per output position * number of output positions
    return nflops * int(np.prod(output_shape))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
