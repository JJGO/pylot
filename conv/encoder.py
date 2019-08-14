from torch import nn

from .block import ConvBlock

class ConvEncoder(nn.Module):

    def __init__(self,
                 in_ch,
                 filters,
                 dims=2,
                 flatten=False,
                 final_pool=True,
                 convs_per_block=2,
                 batch_norm=True):

        super(ConvEncoder, self).__init__()

        self.in_ch = in_ch
        self.filters = filters
        self.flatten = flatten
        self.final_pool = final_pool

        self.downsample = getattr(nn, f"MaxPool{dims}d")(2)

        self.down_blocks = nn.ModuleList()

        for _in_ch, _out_ch in zip([in_ch]+filters[:-1], filters):
            c = ConvBlock(_in_ch, [_out_ch]*convs_per_block,
                          dims=dims,
                          batch_norm=batch_norm)
            self.down_blocks.append(c)


    def forward(self, input):

        x = input

        for i, conv_block in enumerate(self.down_blocks):
            x = conv_block(x)
            if i == len(self.down_blocks) - 1 and not self.final_pool:
                break
            x = self.downsample(x)

        if self.flatten:
            x = x.view(x.size(0), -1)

        return x

