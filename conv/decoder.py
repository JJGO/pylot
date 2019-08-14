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
                 out_activation='Sigmoid' ):

        super(ConvDecoder, self).__init__()

        self.filters = filters

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_blocks = nn.ModuleList()

        for _in_ch, _out_ch in zip([in_ch]+filters[:-1], filters):
            c = ConvBlock(_in_ch, [_out_ch]*convs_per_block,
                          dims=dims,
                          batch_norm=batch_norm)
            self.up_blocks.append(c)

        self.out_conv = ConvBlock(filters[-1], [out_ch], activation=out_activation)

    def forward(self, input):

        x = input

        for conv_block in self.up_blocks:
            x = self.upsample(x)
            x = conv_block(x)

        x = self.out_conv(x)

        return x

