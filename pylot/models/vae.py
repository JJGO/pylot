import math
from dataclasses import dataclass

from typing import List, Dict, Optional, Any, Tuple

import torch
from torch import nn
from ..loss import vae_loss


class VAE(nn.Module):
    def __init__(
        self,
        n_latent: int,
        bottleneck_shape: Tuple[int, ...],
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.bottleneck_shape = bottleneck_shape

        self.encoder = encoder
        self.decoder = decoder

        n_out_enc = math.prod(self.bottleneck_shape)
        self.mu = nn.Linear(n_out_enc, self.n_latent)
        self.logvar = nn.Linear(n_out_enc, self.n_latent)
        self.dec = nn.Linear(self.n_latent, n_out_enc)

    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(enc.size(0), -1)
        return self.mu(enc), self.logvar(enc)

    def decode(self, z):
        x = self.dec(z)
        x = x.view((x.size(0),) + self.bottleneck_shape)
        return self.decoder(x)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.normal(0, 1, size=mu.size()).to(mu.device)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def vae_loss(x_hat, x, mu, logvar, beta=1):
        return vae_loss(x_hat, x, mu, logvar, beta=beta)


class ConvVAE(VAE):
    def __init__(
        self,
        in_size: Tuple[int, ...],
        n_latent: int,
        filters: List[int],
        up_filters: Optional[List[int]] = None,
        enc_kws: Optional[Dict[str, Any]] = None,
        dec_kws: Optional[Dict[str, Any]] = None,
    ):

        from .conv import ConvEncoder, ConvDecoder

        enc_kws = {} if enc_kws is None else enc_kws
        dec_kws = {} if dec_kws is None else dec_kws
        enc_kws["dims"] = len(in_size) - 1
        dec_kws = {**enc_kws, **dec_kws}

        up_filters: List[int] = up_filters if up_filters is not None else filters[::-1]
        in_channels, *in_dims = in_size
        enc_out_size = tuple(
            [filters[-1], *[i // 2 ** (len(filters) - 1) for i in in_dims]]
        )
        encoder = ConvEncoder(in_channels, filters, flatten=True, **enc_kws)
        decoder = ConvDecoder(filters[-1], in_channels, up_filters, **dec_kws)

        super().__init__(n_latent, enc_out_size, encoder, decoder)
