from torch import nn

from .block import ConvBlock


class ConvDecoder(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 filters,
                 dims=2,
                 batch_norm=True,
                 convs_per_block=2,
                 kernel_size=3,
                 first_upsample=True,
                 out_activation='Sigmoid'):

        super(ConvDecoder, self).__init__()

        self.filters = filters
        self.first_upsample = first_upsample

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_blocks = nn.ModuleList()

        for _in_ch, _out_ch in zip([in_ch]+filters[:-1], filters):
            c = ConvBlock(_in_ch, [_out_ch]*convs_per_block,
                          dims=dims,
                          batch_norm=batch_norm,
                          kernel_size=kernel_size)
            self.up_blocks.append(c)

        self.out_conv = ConvBlock(filters[-1], [out_ch],
                                  activation=out_activation,
                                  batch_norm=batch_norm,
                                  kernel_size=kernel_size)

    def forward(self, input):

        x = input

        for i, conv_block in enumerate(self.up_blocks):
            if i > 0 or self.first_upsample:
                x = self.upsample(x)
            x = conv_block(x)

        x = self.out_conv(x)

        return x
