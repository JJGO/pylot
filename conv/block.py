from torch import nn


def ConvBlock(
    inplanes,
    filters,
    dims=2,
    activation="LeakyReLU",
    batch_norm=True,
    kernel_size=3,
    residual=False,  # TODO add residual connections
):

    if activation is not None:
        nonlinearity = getattr(nn, activation)

    conv_fn = getattr(nn, f"Conv{dims}d")
    bn_fn = getattr(nn, f"BatchNorm{dims}d")

    ops = []
    for n_in, n_out in zip([inplanes] + filters, filters):
        conv = conv_fn(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
        )
        ops.append(conv)

        if activation is not None:
            ops.append(nonlinearity())
        if batch_norm:
            ops.append(bn_fn(n_out))
    return nn.Sequential(*ops)

