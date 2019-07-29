import torch
from torch import nn

from .block import ConvBlock

class Unet(nn.Module):


    def __init__(self,
                 in_ch,
                 out_ch,
                 filters,
                 out_activation='Sigmoid',
                 convs_per_block=2):

        super(Unet, self).__init__()

        up_filters = filters[-2::-1]
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.down_blocks = nn.ModuleList()
        for _in_ch, _out_ch in zip([in_ch]+filters[:-1], filters):
            c = ConvBlock(_in_ch, [_out_ch]*convs_per_block)
            self.down_blocks.append(c)


        self.up_blocks = nn.ModuleList()
        for _in_ch, _out_ch, _cat_ch in zip(filters[::-1], up_filters, up_filters):
            c = ConvBlock(_in_ch+_cat_ch, [_out_ch]*convs_per_block)
            self.up_blocks.append(c)

        self.out_conv = ConvBlock(up_filters[-1], [out_ch], activation=out_activation)

    def forward(self, input):

        x = input

        conv_outputs = []

        for i, conv_block in enumerate(self.down_blocks):
            x = conv_block(x)
            if i == len(self.down_blocks) - 1:
                break
            conv_outputs.append(x)
            x = self.downsample(x)
        for i, conv_block in enumerate(self.up_blocks, start=1):
            x = self.upsample(x)
            x = torch.cat([x, conv_outputs[-i]], dim=1)
            x = conv_block(x)


        x = self.out_conv(x)

        return x