"""
Contrastive encoders for temperature-/drift-invariant leak detection.

Two backbone architectures (CNN, Transformer) × two auxiliary heads (AE, Classifier):

CNN variants recycle ``src.anomaly_detection.models.CNNAutoEncoder``.
Transformer variant uses a Transformer-encoder + mirrored decoder, selected via
``encoder_arch='transformer'`` in :func:`build_contrastive_model`.

All models forward to ``(aux_output, z, z_proj)`` where ``aux_output`` is
either the reconstruction (AE) or the class logits (Classifier). Inference
(GMM, Mahalanobis, t-SNE, drift) always uses ``z`` only — the projection
head and auxiliary branch are discarded at deployment time, following
Khosla et al. 2020.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.anomaly_detection.models import CNNAutoEncoder


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with L2-normalised output."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        return F.normalize(h, dim=-1)


class ContrastiveCNNAutoEncoder(nn.Module):
    """
    CNNAutoEncoder + ProjectionHead for supervised contrastive training.

    Architecture:
        Encoder (recycled)  → z          (latent_dim)
        ProjectionHead(z)   → z_proj     (projection_dim, L2-normalised)
        Decoder (recycled)  → x_recon
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: list = None,
        kernel_size: int = 3,
        latent_dim: int = 32,
        projection_dim: int = 64,
        dropout: float = 0.3,
        in_channels: int = 2,
    ):
        super().__init__()
        self.autoencoder = CNNAutoEncoder(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            dropout=dropout,
            in_channels=in_channels,
        )
        self.projection = ProjectionHead(
            in_dim=latent_dim,
            hidden_dim=max(latent_dim, projection_dim),
            out_dim=projection_dim,
        )
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.autoencoder.encode(x)
        x_recon = self.autoencoder.decode(z)
        z_proj = self.projection(z)
        return x_recon, z, z_proj


class ContrastiveCNNClassifier(nn.Module):
    """
    CNN encoder + ProjectionHead + 2-layer classifier head for supervised
    contrastive + cross-entropy training.

    Architecture:
        Encoder (recycled)   → z         (latent_dim)
        ProjectionHead(z)    → z_proj    (projection_dim, L2-normalised)
        Classifier(z)        → logits    (num_classes)

    Encoder is reused from CNNAutoEncoder for direct weight-level comparability
    with the AE variant. The decoder half of CNNAutoEncoder is held in the
    module but never called during forward — its parameters stay static (no
    gradient flow) and can be safely ignored in the optimizer.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: list = None,
        kernel_size: int = 3,
        latent_dim: int = 32,
        projection_dim: int = 64,
        dropout: float = 0.3,
        in_channels: int = 2,
        num_classes: int = 2,
        classifier_hidden: int = None,
    ):
        super().__init__()
        self.autoencoder = CNNAutoEncoder(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            dropout=dropout,
            in_channels=in_channels,
        )
        self.projection = ProjectionHead(
            in_dim=latent_dim,
            hidden_dim=max(latent_dim, projection_dim),
            out_dim=projection_dim,
        )
        hidden = classifier_hidden or max(latent_dim // 2, 32)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        z_proj = self.projection(z)
        return logits, z, z_proj


class TransformerContrastiveAutoEncoder(nn.Module):
    """
    Transformer AutoEncoder + ProjectionHead for supervised contrastive training.

    Input: (B, C, F, W) — same shape as the CNN variant.

    Encoder:
        Flatten (C, F) → token sequence (B, W, C*F)
        Linear projection → (B, W, d_model)
        + learned positional embedding
        N × TransformerEncoderLayer (batch_first=True, norm_first=True)
        mean-pool over W → (B, d_model)
        Linear → z (latent_dim)

    Decoder (mirror, no cross-attention):
        Linear z → W*d_model, reshape → (B, W, d_model)
        + positional embedding
        N × TransformerEncoderLayer
        Linear → (B, W, C*F), reshape → (B, C, F, W)

    ProjectionHead(z) → z_proj (L2-normalised, training only).
    """

    def __init__(
        self,
        in_channels: int,
        input_height: int,
        input_width: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        transformer_dropout: float = 0.1,
        latent_dim: int = 32,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.token_dim = in_channels * input_height
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim

        # --- Encoder ---
        self.enc_proj = nn.Linear(self.token_dim, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, input_width, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=transformer_dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_to_latent = nn.Linear(d_model, latent_dim)

        # --- Projection head (training only) ---
        self.projection = ProjectionHead(
            in_dim=latent_dim,
            hidden_dim=max(latent_dim, projection_dim),
            out_dim=projection_dim,
        )

        # --- Decoder ---
        self.dec_from_latent = nn.Linear(latent_dim, input_width * d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, input_width, d_model) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=transformer_dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.dec_proj = nn.Linear(d_model, self.token_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, W = x.shape
        # (B, C, F, W) → (B, W, C*F)
        tokens = x.reshape(B, C * F, W).transpose(1, 2)
        h = self.enc_proj(tokens) + self.enc_pos[:, :W, :]
        h = self.encoder_transformer(h)
        return self.enc_to_latent(h.mean(dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        W = self.input_width
        h = self.dec_from_latent(z).reshape(B, W, self.d_model)
        h = h + self.dec_pos[:, :W, :]
        h = self.decoder_transformer(h)
        out = self.dec_proj(h)  # (B, W, C*F)
        return out.transpose(1, 2).reshape(B, self.in_channels, self.input_height, W)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        z_proj = self.projection(z)
        return x_recon, z, z_proj


def build_contrastive_model(
    encoder_type: str,
    input_height: int,
    input_width: int,
    channels: list,
    kernel_size: int,
    latent_dim: int,
    projection_dim: int,
    dropout: float,
    in_channels: int,
    num_classes: int = 2,
    encoder_arch: str = 'cnn',
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    transformer_dropout: float = 0.1,
) -> nn.Module:
    """Factory: construct AE or Classifier variant, CNN or Transformer backbone.

    ``encoder_arch`` selects the backbone: 'cnn' (default, unchanged) or
    'transformer'. The Transformer variant is currently only implemented for
    ``encoder_type='autoencoder'``.
    """
    et = encoder_type.lower()
    arch = encoder_arch.lower()

    if arch == 'transformer':
        if et in ('autoencoder', 'ae'):
            return TransformerContrastiveAutoEncoder(
                in_channels=in_channels,
                input_height=input_height,
                input_width=input_width,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                transformer_dropout=transformer_dropout,
                latent_dim=latent_dim,
                projection_dim=projection_dim,
            )
        raise ValueError(
            f"encoder_arch='transformer' is only implemented for encoder_type='autoencoder'. "
            f"Got encoder_type='{encoder_type}'."
        )

    if et in ('autoencoder', 'ae'):
        return ContrastiveCNNAutoEncoder(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            in_channels=in_channels,
        )
    if et in ('classifier', 'clf'):
        return ContrastiveCNNClassifier(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    raise ValueError(
        f"Unknown encoder_type='{encoder_type}'. Expected 'autoencoder' or 'classifier'."
    )
