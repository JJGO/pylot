import torch


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


def G_wgan(model, z_dim, batch_size, device):
    """Generator penalty for wgan. Simply the negated output of the
       discriminator.
    """
    # z = torch.rand((batch_size, z_dim)).to(device)
    # x_fake = G(z)
    x_fake = model.sample(batch_size)
    return -torch.mean(model.D(x_fake))


def D_wgan_gp(model, x_real, z_dim, batch_size, lambda_gp, device):
    """Discriminator penalty for wgan-gp by Gulrajani et al. (https://arxiv.org/abs/1704.00028)"""

    # Loss with real images
    d_out = model.D(x_real)
    d_loss_real = - torch.mean(d_out)

    # Loss with fake images
    # z = torch.rand((batch_size, z_dim)).to(device)
    # x_fake = G(z)
    x_fake = model.sample(batch_size)
    d_out = model.D(x_fake.detach())
    d_loss_fake = torch.mean(d_out)

    # Compute loss for gradient penalty
    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)

    d_out = model.D(x_hat)
    d_loss_gp = gradient_penalty(d_out, x_hat, device)

    return d_loss_real + d_loss_fake + lambda_gp * d_loss_gp
