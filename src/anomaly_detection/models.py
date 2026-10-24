"""
CNN AutoEncoder and Variational AutoEncoder for anomaly-based leakage detection.

The AutoEncoder learns to reconstruct normal (NoFlow) patches.
Leakage patches produce high reconstruction error, serving as anomaly score.

Input shape: (batch, C, H, W) where C=channels, H=features, W=time steps.
Supports single-channel (1, F, W) and dual-channel (2, F/2, W) modes.

Two model types:
- CNNAutoEncoder: Standard AE with deterministic latent space.
- CNNVariationalAutoEncoder: VAE with KL-regularised Gaussian latent space.
"""
from typing import List, Tuple

import torch
import torch.nn as nn


def _build_encoder_conv(
    in_channels: int,
    input_height: int,
    input_width: int,
    channels: List[int],
    kernel_size: int,
) -> Tuple[nn.Sequential, List[Tuple[int, int]], int, int, int]:
    """
    Build shared encoder convolution layers.

    Returns:
        Tuple of (encoder_conv, spatial_sizes, enc_h, enc_w, enc_ch).
    """
    padding = kernel_size // 2
    enc_layers = []
    in_ch = in_channels
    h, w = input_height, input_width
    spatial_sizes = [(h, w)]

    for out_ch in channels:
        enc_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding))
        enc_layers.append(nn.BatchNorm2d(out_ch))
        enc_layers.append(nn.ReLU())
        if h >= 2:
            enc_layers.append(nn.MaxPool2d(2))
            h = h // 2
            w = w // 2
        else:
            enc_layers.append(nn.MaxPool2d((1, 2)))
            w = w // 2
        spatial_sizes.append((h, w))
        in_ch = out_ch

    return nn.Sequential(*enc_layers), spatial_sizes, h, w, channels[-1]


def _build_decoder(
    in_channels: int,
    channels: List[int],
    spatial_sizes: List[Tuple[int, int]],
    kernel_size: int,
) -> Tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList, int]:
    """
    Build shared decoder layers (Upsample + Conv + BN).

    Returns:
        Tuple of (dec_upsample, dec_conv, dec_bn, n_dec_layers).
    """
    padding = kernel_size // 2
    rev_channels = list(reversed(channels))
    target_sizes = list(reversed(spatial_sizes[:-1]))

    dec_upsample = nn.ModuleList()
    dec_conv = nn.ModuleList()
    dec_bn = nn.ModuleList()

    for i in range(len(rev_channels)):
        in_ch_dec = rev_channels[i]
        out_ch_dec = rev_channels[i + 1] if i + 1 < len(rev_channels) else in_channels
        target_h, target_w = target_sizes[i]

        dec_upsample.append(
            nn.Upsample(size=(target_h, target_w), mode='nearest')
        )
        dec_conv.append(
            nn.Conv2d(in_ch_dec, out_ch_dec, kernel_size, padding=padding)
        )
        if i < len(rev_channels) - 1:
            dec_bn.append(nn.BatchNorm2d(out_ch_dec))
        else:
            dec_bn.append(None)

    return dec_upsample, dec_conv, dec_bn, len(rev_channels)


class CNNAutoEncoder(nn.Module):
    """
    2D-CNN AutoEncoder for anomaly detection.

    Encoder compresses patches into a latent vector.
    Decoder reconstructs the original patch from the latent vector.
    Reconstruction error (MSE) serves as anomaly score.

    Architecture:
        Encoder: Conv2d -> BN -> ReLU -> Pool (x N) -> Flatten -> FC -> latent
        Decoder: FC -> Unflatten -> Upsample -> Conv2d -> BN -> ReLU (x N) -> Conv2d
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: List[int] = None,
        kernel_size: int = 5,
        latent_dim: int = 16,
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32]

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # -- Encoder --
        self.encoder_conv, self._spatial_sizes, self._enc_h, self._enc_w, self._enc_ch = \
            _build_encoder_conv(in_channels, input_height, input_width, channels, kernel_size)

        flat_dim = self._enc_ch * self._enc_h * self._enc_w

        self.encoder_fc = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -- Decoder --
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._dec_upsample, self._dec_conv, self._dec_bn, self._n_dec_layers = \
            _build_decoder(in_channels, channels, self._spatial_sizes, kernel_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation (batch, latent_dim)."""
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstructed input."""
        h = self.decoder_fc(z)
        h = h.reshape(h.size(0), self._enc_ch, self._enc_h, self._enc_w)

        for i in range(self._n_dec_layers):
            h = self._dec_upsample[i](h)
            h = self._dec_conv[i](h)
            if self._dec_bn[i] is not None:
                h = self._dec_bn[i](h)
                h = torch.relu(h)

        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Returns:
            Tuple of (reconstructed, latent).
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class CNNVariationalAutoEncoder(nn.Module):
    """
    2D-CNN Variational AutoEncoder for anomaly detection.

    Like CNNAutoEncoder but with a stochastic latent space:
    Encoder outputs mu and log_var, latent is sampled via reparameterisation trick.
    Loss = MSE_reconstruction + beta * KL(q(z|x) || N(0,I)).

    The KL regularisation forces the latent space into a smooth Gaussian,
    clustering normal (NoFlow) data compactly. Anomalies (Flow) land further
    from the learned distribution, improving both reconstruction- and
    latent-distance-based anomaly scores.

    Architecture:
        Encoder: Conv2d -> BN -> ReLU -> Pool (x N) -> Flatten -> FC -> (mu, log_var)
        Latent:  z = mu + std * eps  (reparameterisation trick)
        Decoder: FC -> Unflatten -> Upsample -> Conv2d -> BN -> ReLU (x N) -> Conv2d
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: List[int] = None,
        kernel_size: int = 5,
        latent_dim: int = 16,
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32]

        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # -- Encoder --
        self.encoder_conv, self._spatial_sizes, self._enc_h, self._enc_w, self._enc_ch = \
            _build_encoder_conv(in_channels, input_height, input_width, channels, kernel_size)

        flat_dim = self._enc_ch * self._enc_h * self._enc_w
        self._flat_dim = flat_dim

        # VAE: two heads for mu and log_var instead of single FC
        self.fc_pre = nn.Sequential(
            nn.Linear(flat_dim, flat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(flat_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim // 2, latent_dim)

        # -- Decoder --
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._dec_upsample, self._dec_conv, self._dec_bn, self._n_dec_layers = \
            _build_decoder(in_channels, channels, self._spatial_sizes, kernel_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Returns:
            Tuple of (mu, log_var), each of shape (batch, latent_dim).
        """
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.fc_pre(h)
        return self.fc_mu(h), self.fc_log_var(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z from N(mu, sigma^2) using the reparameterisation trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed input."""
        h = self.decoder_fc(z)
        h = h.reshape(h.size(0), self._enc_ch, self._enc_h, self._enc_w)

        for i in range(self._n_dec_layers):
            h = self._dec_upsample[i](h)
            h = self._dec_conv[i](h)
            if self._dec_bn[i] is not None:
                h = self._dec_bn[i](h)
                h = torch.relu(h)

        return h

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from q(z|x) = N(mu, sigma^2) to prior p(z) = N(0, I).

        Returns scalar (mean over batch).
        """
        return -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, decode.

        Returns:
            Tuple of (reconstructed, z, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var
