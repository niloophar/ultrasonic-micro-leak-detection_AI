"""
Neural Network Models for leakage classification.

Provides MLP, CNN, and Transformer classifiers for binary leakage detection.
Models take patches of shape (1, H, W) or (B, F, W) as input and output a
single logit.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for binary leakage classification.

    Takes a patch of shape (1, H, W) as input, flattens it,
    and passes through fully connected layers.

    Architecture:
        Input: (batch, 1, H, W) -> flatten -> (batch, H*W)
        Hidden layers with ReLU + Dropout
        Output: (batch, 1) raw logit (no activation)

    Loss: BCEWithLogitsLoss(pos_weight=n_neg/n_pos)

    Args:
        input_height: Number of features (H dimension).
        input_width: Number of time steps (W dimension).
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        input_dim = input_height * input_width

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, H, W).

        Returns:
            Raw logits of shape (batch, 1).
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        return self.network(x)


class CNNClassifier(nn.Module):
    """
    2D-CNN for binary leakage classification.

    Treats patches as single-channel 2D images (1, H, W) where H=features,
    W=time steps. Conv layers learn local patterns across feature-time
    neighborhoods, providing implicit regularization through weight sharing.

    Architecture:
        Conv2d(in_channels, c1, k) -> BN -> ReLU -> Pool
        Conv2d(c1, c2, k) -> BN -> ReLU -> AdaptivePool(1,1)
        FC -> Dropout -> FC -> logit

    Args:
        input_height: Number of features (H dimension).
        input_width: Number of time steps (W dimension).
        channels: List of conv channel sizes (default [16, 32]).
        kernel_size: Conv kernel size (default 5).
        fc_dim: Fully connected hidden dim (default 64).
        dropout: Dropout probability (default 0.3).
        in_channels: Number of input channels (default 1).
        dense_input_dim: Number of dense (time-averaged) auxiliary features
            concatenated before the FC classifier. 0 = no dense features.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: List[int] = None,
        kernel_size: int = 5,
        fc_dim: int = 64,
        dropout: float = 0.3,
        in_channels: int = 1,
        dense_input_dim: int = 0,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32]

        padding = kernel_size // 2

        conv_layers = []
        in_ch = in_channels
        h = input_height
        for out_ch in channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding))
            conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU())
            # Only pool dimensions that are large enough (>= 2)
            if h >= 2:
                conv_layers.append(nn.MaxPool2d(2))
                h //= 2
            else:
                # Pool only along time axis (W), keep H unchanged
                conv_layers.append(nn.MaxPool2d((1, 2)))
            in_ch = out_ch

        # Adaptive pool to fixed size regardless of input dims
        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*conv_layers)
        self.dense_input_dim = dense_input_dim
        classifier_input = channels[-1] + dense_input_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor, dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, C, H, W).
            dense: Optional dense features of shape (batch, dense_input_dim).

        Returns:
            Raw logits of shape (batch, 1).
        """
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        if dense is not None and self.dense_input_dim > 0:
            x = torch.cat([x, dense], dim=1)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor, dense: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract feature vectors after CNN pooling (before classifier).

        Returns:
            Feature tensor of shape (batch, channels[-1] + dense_input_dim).
        """
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        if dense is not None and self.dense_input_dim > 0:
            x = torch.cat([x, dense], dim=1)
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer classifier for binary leakage detection (Variant A).

    Treats each time step as a token. Input (B, F, W) is transposed to
    (B, W, F), then projected to d_model dimensions. A standard Transformer
    encoder processes the sequence, followed by mean pooling and a classifier.

    Args:
        n_features: Number of input features (F dimension).
        n_timesteps: Number of time steps (W dimension).
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        d_ff: Feed-forward dimension in Transformer layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        n_timesteps: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.d_model = d_model

        # Project each time-step token (F features) to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_timesteps, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, x: torch.Tensor, dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, F, W).
            dense: Ignored (interface compatibility).

        Returns:
            Raw logits of shape (B, 1).
        """
        # (B, F, W) -> (B, W, F)
        x = x.transpose(1, 2)
        x = self.input_proj(x)  # (B, W, d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)  # (B, W, d_model)
        x = x.mean(dim=1)  # (B, d_model)
        return self.classifier(x)

    def extract_features(
        self, x: torch.Tensor, dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract feature vectors after mean pooling (before classifier).

        Returns:
            Feature tensor of shape (B, d_model).
        """
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        return x.mean(dim=1)


class DualStreamTransformer(nn.Module):
    """
    Dual-stream Transformer with cross-attention (Variant B).

    Splits input features into upstream and downstream halves, processes
    each with self-attention, then applies cross-attention between streams.
    Concatenated mean-pooled representations feed into a classifier.

    Args:
        n_features: Total number of input features (must be even).
        n_timesteps: Number of time steps (W dimension).
        d_model: Per-stream hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of self-attention layers per stream.
        d_ff: Feed-forward dimension in Transformer layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        n_timesteps: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        assert n_features % 2 == 0, (
            f"DualStreamTransformer requires even n_features, got {n_features}"
        )

        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.half_feat = n_features // 2

        # Per-stream input projections
        self.proj_up = nn.Linear(self.half_feat, d_model)
        self.proj_down = nn.Linear(self.half_feat, d_model)

        # Shared positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_timesteps, d_model) * 0.02
        )

        # Shared self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Cross-attention: UP attends to DOWN, DOWN attends to UP
        self.cross_attn_up = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn_down = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm_up = nn.LayerNorm(d_model)
        self.cross_norm_down = nn.LayerNorm(d_model)

        # Classifier on concatenated streams
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _encode_streams(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode upstream and downstream into (B, W, d_model) each."""
        # x: (B, F, W) -> split into up (first half) and down (second half)
        up = x[:, :self.half_feat, :].transpose(1, 2)   # (B, W, F/2)
        down = x[:, self.half_feat:, :].transpose(1, 2)  # (B, W, F/2)

        up = self.proj_up(up) + self.pos_encoding[:, :up.size(1), :]
        down = self.proj_down(down) + self.pos_encoding[:, :down.size(1), :]

        # Self-attention (shared encoder)
        up = self.self_attn(up)
        down = self.self_attn(down)

        # Cross-attention with residual
        up_cross, _ = self.cross_attn_up(up, down, down)
        up = self.cross_norm_up(up + up_cross)

        down_cross, _ = self.cross_attn_down(down, up, up)
        down = self.cross_norm_down(down + down_cross)

        return up, down

    def forward(
        self, x: torch.Tensor, dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, F, W).
            dense: Ignored (interface compatibility).

        Returns:
            Raw logits of shape (B, 1).
        """
        up, down = self._encode_streams(x)
        combined = torch.cat([up.mean(dim=1), down.mean(dim=1)], dim=1)
        return self.classifier(combined)

    def extract_features(
        self, x: torch.Tensor, dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract feature vectors after mean pooling (before classifier).

        Returns:
            Feature tensor of shape (B, 2*d_model).
        """
        up, down = self._encode_streams(x)
        return torch.cat([up.mean(dim=1), down.mean(dim=1)], dim=1)

    def extract_attention_weights(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract cross-attention weight matrices.

        Returns:
            Tuple of (up_to_down_weights, down_to_up_weights),
            each of shape (B, W, W).
        """
        up = x[:, :self.half_feat, :].transpose(1, 2)
        down = x[:, self.half_feat:, :].transpose(1, 2)

        up = self.proj_up(up) + self.pos_encoding[:, :up.size(1), :]
        down = self.proj_down(down) + self.pos_encoding[:, :down.size(1), :]

        up = self.self_attn(up)
        down = self.self_attn(down)

        _, attn_up = self.cross_attn_up(
            up, down, down, need_weights=True, average_attn_weights=True,
        )
        _, attn_down = self.cross_attn_down(
            down, up, up, need_weights=True, average_attn_weights=True,
        )

        return attn_up, attn_down
