import torch
import torch.nn.functional as F

def kld_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld

def vae_loss(x_hat, x, mu, logvar, beta=1,
             recon_loss=F.binary_cross_entropy,
             return_all=False):

    fx = x.view(x.size(0), -1)
    fx_hat = x_hat.view(x_hat.size(0), -1)
    bce = recon_loss(fx_hat, fx, reduction='sum')

    kld = kld_loss(mu, logvar)

    loss = bce + beta * kld

    if return_all:
        return loss, bce, kld
    else:
        return loss
