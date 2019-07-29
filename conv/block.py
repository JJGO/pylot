from torch import nn

def ConvBlock(inplanes,
              filters,
              activation='ReLU',
              batch_norm=True,
              kernel_size=3):

    nonlinearity = getattr(nn, activation)

    ops = []
    for i, (n_in, n_out) in enumerate(zip([inplanes]+filters, filters)):
        conv = nn.Conv2d(n_in, n_out,
                         kernel_size=kernel_size,
                         padding=kernel_size//2,
                         padding_mode='reflection')
        ops.append(conv)

        if batch_norm:
            bn = nn.BatchNorm2d(n_out)
            ops.append(bn)

        ops.append(nonlinearity())
    return nn.Sequential(*ops)
