"""Defines reconstruction loss."""

import torch


def reconstruction_loss(x, x_recon):
    """Compute the reconstruction loss between two sequences.

    This is computed as the mean square errors on the joints'
    positions in 3D.

    Parameters
    ----------
    x :             array
                    Shape = [batch_size, seq_len, input_dim]
                    Input batch of sequences.
    x_recon :       array
                    Shape = [batch_size, seq_len, input_dim]
                    Batch of reconstructed sequences.

    Returns
    ----------
    recon_loss :    array
                    Shape = [batch_size, 1]
                    Reconstruction loss for each sequence
                    in batch.
    """
    assert x.ndim == x_recon.ndim == 3
    batch_size = x.shape[0]
    recon_loss = (x - x_recon) ** 2

    recon_loss = torch.mean(recon_loss, axis=(1, 2))

    assert recon_loss.shape == (batch_size,)
    return recon_loss


def kld(q_param):
    """Compute KL-divergence.
    The KL is defined as:
    KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                = -E[log p(z) - log q(z)]
    Formula in Keras was:
    # -0.5*K.mean(K.sum(1 + auto_log_var -
    # K.square(auto_mean) - K.exp(auto_log_var), axis=-1))
    Parameters
    ----------
    z : array-like
        Sample from q-distribuion
    q_param : tuple
        (mu, log_var) of the q-distribution
    Returns
    -------
    kl : KL(q||p)
    """
    z_mean, z_logvar = q_param
    kl = -(torch.sum(1 + z_logvar - torch.square(z_mean) - z_logvar.exp(), axis=-1))
    return kl
