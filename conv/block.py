from torch import nn

def ConvBlock(inplanes,
              filters,
              dims=2,
              activation='LeakyReLU',  #'ReLU',
              batch_norm=True,
              kernel_size=3):

    if activation is not None:
        nonlinearity = getattr(nn, activation)

    conv_fn = getattr(nn, f"Conv{dims}d")
    bn_fn = getattr(nn, f"BatchNorm{dims}d")

    ops = []
    for i, (n_in, n_out) in enumerate(zip([inplanes]+filters, filters)):
        conv = conv_fn(n_in, n_out,
                       kernel_size=kernel_size,
                       padding=kernel_size//2,
                       padding_mode='zeros')
        ops.append(conv)

        if batch_norm:
            bn = bn_fn(n_out)
            ops.append(bn)
        if activation is not None:
            ops.append(nonlinearity())
    return nn.Sequential(*ops)
