import torch
from torch import nn
from ..loss import vae_loss


class VAE(nn.Module):
    def __init__(self, n_latent, n_out_enc, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.n_latent = n_latent
        self.n_out_enc = n_out_enc
        self.mu = nn.Linear(n_out_enc, n_latent)
        self.logvar = nn.Linear(n_out_enc, n_latent)
        self.dec = nn.Linear(n_latent, n_out_enc)

    def encode(self, x):
        enc = self.encoder(x)
        self._last_size = enc.size()
        enc = enc.view(enc.size(0), -1)
        return self.mu(enc), self.logvar(enc)

    def decode(self, z, shape=None):
        x = self.dec(z)

        if shape is None:
            x = x.view(*self._last_size)
            self._last_size = None
        else:
            x = x.view(*shape)

        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z), mu, logvar
        return output

    @staticmethod
    def vae_loss(x_hat, x, mu, logvar, beta=1):
        return vae_loss(x_hat, x, mu, logvar, beta=beta)
