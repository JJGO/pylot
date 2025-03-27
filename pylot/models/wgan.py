import torch
from torch import nn

# from ..loss.wgan import gradient_penalty, G_wgan, D_wgan_gp

from .encoder import ConvEncoder
from .decoder import ConvDecoder


class ConvWGAN(nn.Module):

    def __init__(self, G_latent, G_filters, D_filters, height, width, out_ch=1, batch_norm=False):

        super(ConvWGAN, self).__init__()

        self.G_latent = G_latent
        self.G_filters = G_filters
        self.D_filters = D_filters
        self.height = height
        self.width = width
        self.out_ch = out_ch

        # If they aren't a multiple of a power of two
        # we can't produce them
        ratio = 2**(len(G_filters)-1)
        assert height % ratio == 0
        assert width % ratio == 0

        self.init_h = height // ratio
        self.init_w = width // ratio

        assert G_latent % (self.init_h * self.init_w) == 0

        self.init_c = G_latent // (self.init_h * self.init_w)

        self.G = ConvDecoder(self.init_c, self.out_ch, self.G_filters,
                             batch_norm=batch_norm, convs_per_block=1,
                             out_activation=None, first_upsample=False,
                             kernel_size=5)
        self.D = ConvEncoder(self.out_ch, D_filters, batch_norm=batch_norm,
                             convs_per_block=1, final_pool=False,
                             kernel_size=5)

    def sample(self, batch_size=1):
        z = torch.randn((batch_size,
                         self.init_c,
                         self.init_h,
                         self.init_w,))
        z = z.to(next(self.G.parameters()).device)
        return self.G(z)

    @property
    def Generator(self):
        return self.G

    @property
    def Discriminator(self):
        return self.D

    @property
    def input_size(self):
        return (self.init_c, self.init_h, self.init_w)

