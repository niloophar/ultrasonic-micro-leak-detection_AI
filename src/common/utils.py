"""
Shared utility functions for classification and anomaly detection pipelines.
"""
from typing import List

import numpy as np


def detect_n_stats(feature_names: List[str]) -> int:
    """Detect how many downsample statistics are present from feature names.

    Convention: base features have no '__' suffix (mean).
    Additional stats are suffixed with '__std', '__min', etc.

    Returns:
        Number of stat groups (1 = mean only, 2 = mean + std, etc.).
    """
    base = [n for n in feature_names if '__' not in n]
    suffixed = [n for n in feature_names if '__' in n]
    if not base:
        return 1
    n_stats = 1 + len(suffixed) // len(base) if base else 1
    return max(1, n_stats)


def reshape_to_channels(
    arr: np.ndarray,
    n_feat: int,
    w: int,
    n_stats: int = 1,
) -> tuple:
    """Reshape (N, F, W) to (N, C, H, W) for dual-channel CNN input.

    Channel layout:
    - n_stats=1: (N, 2, F/2, W) — up | down
    - n_stats=2: (N, 4, F/4, W) — mean_up | mean_down | std_up | std_down

    Args:
        arr: Input array of shape (N, F, W).
        n_feat: Number of features F.
        w: Patch width W.
        n_stats: Number of downsample statistics (1=mean, 2=mean+std).

    Returns:
        Tuple of (reshaped_array, n_in_channels, h) where h = features per channel.
    """
    n_base = n_feat // n_stats
    half_base = n_base // 2
    n_channels = 2 * n_stats

    if len(arr) == 0:
        return arr.reshape(0, n_channels, half_base, w), n_channels, half_base

    channels = []
    for s in range(n_stats):
        offset = s * n_base
        channels.append(arr[:, offset:offset + half_base, :])          # up
        channels.append(arr[:, offset + half_base:offset + n_base, :])  # down
    return np.stack(channels, axis=1), n_channels, half_base
