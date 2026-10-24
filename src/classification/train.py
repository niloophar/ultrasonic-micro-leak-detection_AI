"""
Training and Evaluation Module - Milestone C4.

Provides:
- Loading pre-built patches from the data preparation step
- Training loop with early stopping and learning rate scheduling
- Evaluation with accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- Plot generation: loss curves, confusion matrix, ROC curve
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .models import (
    MLPClassifier, CNNClassifier,
    TransformerClassifier, DualStreamTransformer,
)
from src.common.utils import detect_n_stats, reshape_to_channels


# Prep-time scaling modes that fully standardize the data.
# When detected in metadata, train.py skips its own StandardScaler.
_PREP_FULL_STANDARDIZE_MODES = ("noflow_global", "noflow_per_series")


def _prep_scaling_mode(metadata_by_id: Dict[int, dict]) -> str:
    """Return the prep-time scaling mode recorded in metadata, or 'none'."""
    if not metadata_by_id:
        return "none"
    any_meta = next(iter(metadata_by_id.values()), None)
    if any_meta is None:
        return "none"
    return any_meta.get("prep_scaling_mode", "none")


def _skip_train_scaler(prep_mode: str) -> bool:
    """True if the train-time StandardScaler should be skipped."""
    return prep_mode in _PREP_FULL_STANDARDIZE_MODES


def _scaled_subdir(base_dir: Path, metadata_by_id: Dict[int, dict]) -> Path:
    """Append a 'scaling_<mode>' subdirectory if prep-time scaling is active.

    For the default mode 'none', the base directory is returned unchanged —
    keeping backwards-compatible paths. For any other mode, results land in
    a dedicated subdirectory so variants can coexist without overwriting.
    """
    mode = _prep_scaling_mode(metadata_by_id)
    if mode == "none":
        return base_dir
    return base_dir / f"scaling_{mode}"


# ---------------------------------------------------------------------------
# Load prepared data
# ---------------------------------------------------------------------------

def _load_prepared_data(
    segment_ids: List[int],
    prepared_base: Path,
    metadata_by_id: Dict[int, dict],
    selected_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[str]]]:
    """
    Load pre-built .npy patch files for a list of segment IDs.

    Args:
        segment_ids: List of segment IDs to load.
        prepared_base: Path to prepared/ directory.
        metadata_by_id: Mapping segment_id -> preparation metadata dict.
        selected_features: If given, only keep these feature columns.
            Columns not present are silently skipped.

    Returns:
        Tuple of:
        - patches: Shape (n_patches, n_features, patch_width)
        - labels: Shape (n_patches,)
        - series_per_patch: Series name for each patch
        - feature_names: Feature column names (after selection)
    """
    all_patches = []
    all_labels = []
    all_series = []
    feature_names: Optional[List[str]] = None
    feature_indices: Optional[List[int]] = None

    for seg_id in tqdm(segment_ids, desc="    Loading patches", unit="seg"):
        meta = metadata_by_id.get(seg_id)
        if meta is None or 'patch_file' not in meta:
            continue

        patch_path = prepared_base / meta['series'] / meta['patch_file']
        if not patch_path.exists():
            continue

        patches = np.load(patch_path)  # Shape: (n_patches, n_features, patch_width)
        if len(patches) == 0:
            continue

        # Apply feature selection on first successful load
        if feature_names is None:
            all_names = meta.get('feature_names', [])
            if selected_features is not None and all_names:
                feature_indices = [
                    i for i, name in enumerate(all_names)
                    if name in selected_features
                ]
                if not feature_indices:
                    continue
                feature_names = [all_names[i] for i in feature_indices]
            else:
                feature_names = all_names

        # Filter features if needed
        if feature_indices is not None:
            patches = patches[:, feature_indices, :]

        all_patches.append(patches)
        all_labels.extend([meta['label']] * len(patches))
        all_series.extend([meta['series']] * len(patches))

    if not all_patches:
        pw = next(iter(metadata_by_id.values()), {}).get('patch_width', 32)
        return np.empty((0, 0, pw)), np.array([]), [], feature_names

    return (
        np.concatenate(all_patches, axis=0),
        np.array(all_labels),
        all_series,
        feature_names,
    )


def _compute_pos_weight(labels: np.ndarray) -> float:
    """Compute pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


# ---------------------------------------------------------------------------
# Feature importance helpers (XGBoost)
# ---------------------------------------------------------------------------

def _aggregate_feature_importance(
    raw_importances: np.ndarray,
    feature_names: List[str],
    patch_width: int,
    n_dense: int = 0,
) -> Dict[str, float]:
    """
    Map flattened feature importances back to original feature names.

    XGBoost trains on (N, F_cnn*W + n_dense) flattened input. Each CNN feature
    occupies W consecutive columns; each dense feature occupies 1 column.
    Sum importances over the W time steps per CNN feature, keep dense as-is,
    then normalise to sum=1.

    Args:
        raw_importances: Shape (F_cnn*W + n_dense,) importance vector.
        feature_names: Original feature column names (CNN then dense).
        patch_width: Number of time steps per patch (W).
        n_dense: Number of dense (time-averaged) features at the end.

    Returns:
        Dict mapping feature name to aggregated importance.
    """
    n_cnn = len(feature_names) - n_dense
    cnn_imp = raw_importances[:n_cnn * patch_width].reshape(n_cnn, patch_width)
    per_feature = cnn_imp.sum(axis=1)

    if n_dense > 0:
        dense_imp = raw_importances[n_cnn * patch_width:n_cnn * patch_width + n_dense]
        per_feature = np.concatenate([per_feature, dense_imp])

    total = per_feature.sum()
    if total > 0:
        per_feature = per_feature / total
    return {name: float(val) for name, val in zip(feature_names, per_feature)}


def _permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    patch_width: int,
    n_repeats: int = 5,
    n_dense: int = 0,
) -> Dict[str, float]:
    """
    Compute permutation importance by shuffling each original feature's
    columns and measuring accuracy drop.

    Args:
        model: Trained XGBClassifier with predict method.
        X: Flattened test data, shape (N, F_cnn*W + n_dense).
        y: True labels, shape (N,).
        feature_names: Original feature column names (CNN then dense).
        patch_width: Number of time steps per patch (W).
        n_repeats: Number of shuffle repetitions to average.
        n_dense: Number of dense (time-averaged) features at the end.

    Returns:
        Dict mapping feature name to mean accuracy drop.
    """
    rng = np.random.default_rng(42)
    baseline_acc = float((model.predict(X) == y).mean())

    n_cnn = len(feature_names) - n_dense
    result = {}

    for f_idx in range(len(feature_names)):
        if f_idx < n_cnn:
            col_start = f_idx * patch_width
            col_end = (f_idx + 1) * patch_width
        else:
            # Dense feature: single column after all CNN columns
            dense_offset = n_cnn * patch_width + (f_idx - n_cnn)
            col_start = dense_offset
            col_end = dense_offset + 1

        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X))
            X_perm[:, col_start:col_end] = X_perm[perm_idx, col_start:col_end]
            perm_acc = float((model.predict(X_perm) == y).mean())
            drops.append(baseline_acc - perm_acc)

        result[feature_names[f_idx]] = float(np.mean(drops))

    return result


def _save_importance_plot(
    importances: Dict[str, float],
    save_dir: Path,
    title: str,
    filename: str,
    top_k: int = 20,
) -> None:
    """Save a horizontal bar chart of feature importances."""
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:top_k]
    names = [item[0] for item in reversed(top)]
    values = [item[1] for item in reversed(top)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    ax.barh(names, values, color='#2196F3')
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    fig.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_importance_comparison_plot(
    overall: Dict[str, float],
    per_series: Dict[str, Dict[str, float]],
    save_dir: Path,
    top_k: int = 15,
) -> None:
    """Save grouped horizontal bar chart comparing feature importance across series."""
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    sorted_features = sorted(overall.items(), key=lambda x: x[1], reverse=True)
    top_features = [item[0] for item in sorted_features[:top_k]]

    all_labels = ['Overall'] + sorted(per_series.keys())
    all_dicts = [overall] + [per_series[s] for s in sorted(per_series.keys())]

    n_groups = len(all_labels)
    y_pos = np.arange(len(top_features))
    bar_height = 0.8 / n_groups

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 3)))

    fig, ax = plt.subplots(figsize=(12, max(5, len(top_features) * 0.5)))

    for g_idx, (label, imp_dict) in enumerate(zip(all_labels, all_dicts)):
        values = [imp_dict.get(f, 0.0) for f in reversed(top_features)]
        offsets = y_pos + g_idx * bar_height
        ax.barh(offsets, values, height=bar_height, label=label,
                color=colors[g_idx])

    ax.set_yticks(y_pos + bar_height * (n_groups - 1) / 2)
    ax.set_yticklabels(list(reversed(top_features)))
    ax.set_xlabel('Permutation Importance (accuracy drop)')
    ax.set_title('Feature Importance Comparison Across Series')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    fig.savefig(save_dir / 'feature_importance_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_importance_stability_plot(
    per_series_importance: Dict[str, Dict[str, float]],
    save_dir: Path,
    top_n: int = 10,
) -> None:
    """Save a line plot showing feature importance stability across series.

    x-axis: series names sorted chronologically (parse DD.MM.YYYY).
    y-axis: permutation importance (accuracy drop).
    One line per top-N feature (ranked by mean importance across series).
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from datetime import datetime

    if not per_series_importance:
        return

    # Sort series chronologically by parsing DD.MM.YYYY
    def _parse_date(name: str) -> datetime:
        try:
            return datetime.strptime(name, "%d.%m.%Y")
        except ValueError:
            return datetime.max

    sorted_series = sorted(per_series_importance.keys(), key=_parse_date)

    # Compute mean importance per feature across all series
    all_features: Dict[str, List[float]] = {}
    for sname in sorted_series:
        imp = per_series_importance[sname]
        for feat, val in imp.items():
            all_features.setdefault(feat, []).append(val)

    mean_importance = {
        feat: float(np.mean(vals)) for feat, vals in all_features.items()
    }
    top_features = sorted(
        mean_importance.items(), key=lambda x: x[1], reverse=True
    )[:top_n]
    top_names = [f[0] for f in top_features]

    # Build data matrix
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(top_names), 10)))

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_series) * 1.5), 6))

    for i, feat in enumerate(top_names):
        values = [
            per_series_importance[s].get(feat, 0.0) for s in sorted_series
        ]
        ax.plot(
            sorted_series, values,
            marker='o', linewidth=2, markersize=6,
            label=feat, color=colors[i % len(colors)],
        )

    ax.set_xlabel('Series')
    ax.set_ylabel('Permutation Importance (accuracy drop)')
    ax.set_title(f'Feature Importance Stability (top {len(top_names)})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    fig.savefig(save_dir / 'feature_importance_stability.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_model(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
    model_name: str = 'mlp',
) -> Dict:
    """
    Train MLP/CNN/Transformer classifier and evaluate on test set.

    Loads splits.json and prepared patch metadata, loads pre-built patches,
    fits a scaler on training data, and trains the model.

    Args:
        config: Full config dict (must contain 'classification').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot_enabled: Whether to create plots.
        only_files: Unused.
        max_workers: Unused.
        model_name: Which model config to use ('mlp' or 'transformer').

    Returns:
        Dict with training results and metrics.
    """
    # Determine config source based on model_name
    if model_name == 'transformer':
        trans_config = config['classification'].get('transformer', {})
        model_type = 'transformer'
        transformer_variant = trans_config.get('transformer_variant', 'A')
        active_config = trans_config
        output_root = trans_config.get('output_root', './prepared')
        dropout = trans_config.get('dropout', 0.3)
        lr = trans_config.get('lr', 1e-3)
        batch_size = trans_config.get('batch_size', 32)
        epochs = trans_config.get('epochs', 100)
        patience = trans_config.get('patience', 10)
        scaler_type = trans_config.get('scaler', 'standard')
        selected_features = trans_config.get('selected_features', None)
        dense_features = []
        # Transformer-specific params
        d_model = trans_config.get('d_model', 64)
        n_heads = trans_config.get('n_heads', 4)
        n_layers = trans_config.get('n_layers', 2)
        d_ff = trans_config.get('d_ff', 128)
        # Not used for transformer
        hidden_dims = None
        cnn_channels = None
        cnn_kernel_size = None
        cnn_fc_dim = None
        cnn_channel_mode = 'single'
    else:
        mlp_config = config['classification'].get('mlp', {})
        active_config = mlp_config
        output_root = mlp_config.get('output_root', './prepared')
        model_type = mlp_config.get('model_type', 'mlp')
        hidden_dims = mlp_config.get('hidden_dims', [128, 64])
        dropout = mlp_config.get('dropout', 0.3)
        lr = mlp_config.get('lr', 1e-3)
        batch_size = mlp_config.get('batch_size', 32)
        epochs = mlp_config.get('epochs', 100)
        patience = mlp_config.get('patience', 10)
        cnn_channels = mlp_config.get('cnn_channels', [16, 32])
        cnn_kernel_size = mlp_config.get('cnn_kernel_size', 5)
        cnn_fc_dim = mlp_config.get('cnn_fc_dim', 64)
        cnn_channel_mode = mlp_config.get('cnn_channel_mode', 'single')
        scaler_type = mlp_config.get('scaler', 'standard')
        selected_features = mlp_config.get('selected_features', None)
        dense_features = mlp_config.get('dense_features', None) or []
        transformer_variant = None
        d_model = n_heads = n_layers = d_ff = None

    if model_type == 'transformer':
        title = f"TRAIN TRANSFORMER CLASSIFIER (Variant {transformer_variant})"
    elif model_type == "cnn":
        title = "TRAIN CNN CLASSIFIER"
    else:
        title = "TRAIN MLP CLASSIFIER"
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Model type:        {model_type}")
    if model_type == 'transformer':
        print(f"Variant:           {transformer_variant}")
        print(f"d_model:           {d_model}")
        print(f"n_heads:           {n_heads}")
        print(f"n_layers:          {n_layers}")
        print(f"d_ff:              {d_ff}")
    elif model_type == "cnn":
        print(f"CNN channels:      {cnn_channels}")
        print(f"CNN kernel size:   {cnn_kernel_size}")
        print(f"CNN FC dim:        {cnn_fc_dim}")
        print(f"CNN channel mode:  {cnn_channel_mode}")
    else:
        print(f"Hidden dims:       {hidden_dims}")
    print(f"Dropout:           {dropout}")
    print(f"LR:                {lr}")
    print(f"Batch size:        {batch_size}")
    print(f"Max epochs:        {epochs}")
    print(f"Patience:          {patience}")
    print(f"Scaler:            {scaler_type}")
    if selected_features:
        print(f"Selected features: {len(selected_features)} columns")
    if dense_features:
        print(f"Dense features:    {len(dense_features)} columns")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:            {device}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Load splits.json and prepared metadata
    # ---------------------------------------------------------------
    class_base = data_path / output_root
    splits_path = class_base / 'splits.json'
    if not splits_path.exists():
        print(f"[ERROR] splits.json not found: {splits_path}")
        print("Run 'splitting' step first.")
        return {}

    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_dir_name = active_config.get('prepared_dir', 'prepared')
    prepared_base = class_base / prepared_dir_name
    prep_meta_path = prepared_base / 'metadata.json'
    if not prep_meta_path.exists():
        print(f"[ERROR] Prepared metadata not found: {prep_meta_path}")
        print("Run 'prepare' step first.")
        return {}

    with open(prep_meta_path, 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}

    # Results directory — optionally suffixed by prep-time scaling mode
    # so that scaler variants don't overwrite each other.
    results_dir = _scaled_subdir(
        data_path / output_root / 'results', metadata_by_id,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    # ---------------------------------------------------------------
    # Load pre-built patches
    # ---------------------------------------------------------------
    # Merge CNN + dense features for loading, then split by column index
    all_features = list(dict.fromkeys(
        (selected_features or []) + dense_features
    )) or None

    print("\n--- LOADING TRAINING DATA ---")
    train_patches, train_labels, train_series_per_patch, loaded_feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, all_features,
    )

    if len(train_patches) == 0:
        print("[ERROR] No training patches could be loaded.")
        return {}

    print(f"  Train patches: {train_patches.shape}")

    print("\n--- LOADING VALIDATION DATA ---")
    val_patches, val_labels, val_series_per_patch, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, all_features,
    )
    print(f"  Val patches: {val_patches.shape}")

    print("\n--- LOADING TEST DATA ---")
    test_patches, test_labels, test_series_per_patch, _ = _load_prepared_data(
        test_ids, prepared_base, metadata_by_id, all_features,
    )
    print(f"  Test patches: {test_patches.shape}")

    # ---------------------------------------------------------------
    # Split into CNN features and dense features
    # ---------------------------------------------------------------
    loaded_names = loaded_feature_names or []
    if dense_features and loaded_names:
        cnn_feature_names = [f for f in loaded_names if f not in dense_features]
        dense_indices = [i for i, f in enumerate(loaded_names) if f in dense_features]
        cnn_indices = [i for i, f in enumerate(loaded_names) if f not in dense_features]
        n_dense = len(dense_indices)
        print(f"\n  CNN features ({len(cnn_indices)}): {cnn_feature_names}")
        print(f"  Dense features ({n_dense}): {[loaded_names[i] for i in dense_indices]}")

        def _split_dense(patches):
            if len(patches) == 0:
                return (patches[:, cnn_indices, :] if cnn_indices else patches,
                        np.empty((0, n_dense)))
            dense_arr = patches[:, dense_indices, :].mean(axis=2)  # (N, n_dense)
            cnn_arr = patches[:, cnn_indices, :] if cnn_indices else patches
            return cnn_arr, dense_arr

        train_patches, train_dense = _split_dense(train_patches)
        val_patches, val_dense = _split_dense(val_patches)
        test_patches, test_dense = _split_dense(test_patches)
    else:
        n_dense = 0
        cnn_feature_names = loaded_names
        train_dense = val_dense = test_dense = None

    # ---------------------------------------------------------------
    # Fit scaler on training data (skipped if prep-time scaling active)
    # ---------------------------------------------------------------
    print("\n--- SCALING ---")
    n_train, n_feat, w = train_patches.shape
    prep_mode = _prep_scaling_mode(metadata_by_id)

    if _skip_train_scaler(prep_mode):
        print(f"  Prep-time scaling active (mode: {prep_mode}) — train-time scaler skipped")
        def _scale(patches_arr):
            return patches_arr
    else:
        # Reshape for scaler: (n_patches * patch_width, n_features)
        train_flat = train_patches.transpose(0, 2, 1).reshape(-1, n_feat)

        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(train_flat)
        print(f"  Scaler fitted on {len(train_flat)} samples x {n_feat} features")

        def _scale(patches_arr):
            if len(patches_arr) == 0:
                return patches_arr
            n, f, w_ = patches_arr.shape
            flat = patches_arr.transpose(0, 2, 1).reshape(-1, f)
            scaled = scaler.transform(flat)
            return scaled.reshape(n, w_, f).transpose(0, 2, 1)

    train_scaled = _scale(train_patches)
    val_scaled = _scale(val_patches)
    test_scaled = _scale(test_patches)

    # Scale dense features separately
    if n_dense > 0 and train_dense is not None:
        if scaler_type == 'minmax':
            dense_scaler = MinMaxScaler()
        else:
            dense_scaler = StandardScaler()
        dense_scaler.fit(train_dense)
        train_dense = dense_scaler.transform(train_dense)
        if len(val_dense) > 0:
            val_dense = dense_scaler.transform(val_dense)
        if len(test_dense) > 0:
            test_dense = dense_scaler.transform(test_dense)
        print(f"  Dense scaler fitted on {len(train_dense)} samples x {n_dense} features")

    # Reshape for model input
    n_stats = detect_n_stats(cnn_feature_names) if cnn_feature_names else 1
    if model_type == 'transformer':
        # Transformer takes (B, F, W) directly — no channel dim
        n_in_channels = 1  # unused, but set for consistency
        print(f"  Transformer input: ({n_feat}, {w}) — no channel reshape")
    elif model_type == 'cnn' and cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        train_scaled, n_in_channels, h = reshape_to_channels(
            train_scaled, n_feat, w, n_stats)
        val_scaled, _, _ = reshape_to_channels(val_scaled, n_feat, w, n_stats)
        test_scaled, _, _ = reshape_to_channels(test_scaled, n_feat, w, n_stats)
        print(f"  Dual-channel reshape: ({n_feat},) -> ({n_in_channels}, {h})")
    else:
        n_in_channels = 1
        train_scaled = train_scaled[:, np.newaxis, :, :]
        if len(val_scaled) > 0:
            val_scaled = val_scaled[:, np.newaxis, :, :]
        else:
            val_scaled = val_scaled.reshape(0, 1, n_feat, w)
        if len(test_scaled) > 0:
            test_scaled = test_scaled[:, np.newaxis, :, :]
        else:
            test_scaled = test_scaled.reshape(0, 1, n_feat, w)

    # ---------------------------------------------------------------
    # Create DataLoaders from in-memory tensors
    # ---------------------------------------------------------------
    def _make_loader(patches_arr, labels_arr, dense_arr, shuffle):
        if len(patches_arr) == 0:
            return None
        t_patches = torch.tensor(patches_arr, dtype=torch.float32)
        t_labels = torch.tensor(labels_arr, dtype=torch.float32)
        if dense_arr is not None and len(dense_arr) > 0:
            t_dense = torch.tensor(dense_arr, dtype=torch.float32)
            ds = TensorDataset(t_patches, t_dense, t_labels)
        else:
            ds = TensorDataset(t_patches, t_labels)
        g = torch.Generator().manual_seed(42) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=torch.cuda.is_available(),
                          generator=g)

    train_loader = _make_loader(train_scaled, train_labels, train_dense, shuffle=True)
    val_loader = _make_loader(val_scaled, val_labels, val_dense, shuffle=False)
    test_loader = _make_loader(test_scaled, test_labels, test_dense, shuffle=False)

    if train_loader is None:
        print("[ERROR] No training data found.")
        return {}

    # Infer input dimensions from first batch
    sample_batch = next(iter(train_loader))
    if model_type == 'transformer':
        _, h, w_dim = sample_batch[0].shape
        c_dim = None
        print(f"\nInput shape: ({h}, {w_dim})")
    else:
        _, c_dim, h, w_dim = sample_batch[0].shape
        print(f"\nInput shape: ({c_dim}, {h}, {w_dim})")
    if n_dense > 0:
        print(f"Dense dim:   {n_dense}")
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")
    if test_loader:
        print(f"Test batches:  {len(test_loader)}")

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    # Model
    if model_type == 'transformer':
        if transformer_variant == 'B':
            model = DualStreamTransformer(
                h, w_dim, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ).to(device)
        else:
            model = TransformerClassifier(
                h, w_dim, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ).to(device)
    elif model_type == "cnn":
        model = CNNClassifier(
            h, w_dim, channels=cnn_channels,
            kernel_size=cnn_kernel_size, fc_dim=cnn_fc_dim,
            dropout=dropout, in_channels=n_in_channels,
            dense_input_dim=n_dense,
        ).to(device)
    else:
        model = MLPClassifier(h, w_dim, hidden_dims, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params:  {n_params:,}")

    # Loss with class weighting
    pos_weight_cfg = active_config.get('pos_weight', 'auto')
    if pos_weight_cfg == 'auto' or pos_weight_cfg is None:
        pos_weight = _compute_pos_weight(train_labels)
    else:
        pos_weight = float(pos_weight_cfg)
    print(f"Pos weight:    {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    def _unpack_batch(batch):
        """Unpack batch into (patches, dense_or_None, labels)."""
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        return batch[0], None, batch[1]

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            patches, dense_batch, labels = _unpack_batch(batch)
            patches = patches.to(device)
            labels = labels.to(device).unsqueeze(1)
            if dense_batch is not None:
                dense_batch = dense_batch.to(device)

            optimizer.zero_grad()
            logits = model(patches, dense=dense_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss = avg_train_loss  # fallback
        if val_loader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, dense_batch, labels = _unpack_batch(batch)
                    patches = patches.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    if dense_batch is not None:
                        dense_batch = dense_batch.to(device)
                    logits = model(patches, dense=dense_batch)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    n_val += 1
            avg_val_loss = val_loss / max(n_val, 1)

        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"LR: {current_lr:.1e}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(best: epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Save model
    model_path = results_dir / 'mlp_best.pt'
    torch.save(best_state or model.state_dict(), model_path)
    print(f"\nBest model saved: {model_path}")

    # Evaluate on validation set
    if val_loader:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON VALIDATION SET")
        print(f"{'=' * 60}")
        val_metrics = _evaluate(model, val_loader, device)
        _print_metrics(val_metrics)

    # Evaluate on test set
    metrics = {}
    per_series_metrics = {}
    if test_loader:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON TEST SET (overall)")
        print(f"{'=' * 60}")
        metrics = _evaluate(model, test_loader, device)
        _print_metrics(metrics)

        # Per-series evaluation
        per_series_metrics = _evaluate_per_series(
            model, test_scaled, test_labels, test_series_per_patch, device,
            test_dense=test_dense,
        )
        for sname, smetrics in per_series_metrics.items():
            print(f"\n{'=' * 60}")
            print(f"EVALUATION ON TEST SET \u2014 {sname}")
            print(f"{'=' * 60}")
            _print_metrics(smetrics)

    elif val_loader:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON VALIDATION SET (no test data available)")
        print(f"{'=' * 60}")
        metrics = _evaluate(model, val_loader, device)
        _print_metrics(metrics)

    # Generate plots — always save to img_path / classification (optionally
    # in a scaling_<mode> subdirectory so variants don't overwrite each other)
    plot_dir = _scaled_subdir(img_path / 'classification', metadata_by_id)
    plot_dir.mkdir(parents=True, exist_ok=True)
    display_model_name = model_type.upper()
    if model_type == 'transformer':
        display_model_name = f"TRANSFORMER_{transformer_variant}"
    _save_plots(
        train_losses, val_losses, metrics, plot_dir,
        model_name=display_model_name, model_config=active_config,
    )
    for sname, smetrics in per_series_metrics.items():
        _save_plots(
            train_losses, val_losses, smetrics, plot_dir,
            model_name=display_model_name, model_config=active_config,
            series_name=sname,
        )

    # Cross-attention heatmaps (Variant B only)
    if model_type == 'transformer' and transformer_variant == 'B' and test_loader:
        _save_attention_heatmaps(
            model, test_scaled, test_labels, test_series_per_patch,
            device, plot_dir,
        )

    # Logit distribution histograms (train, val, test per series)
    logit_splits = {}
    if len(train_scaled) > 0:
        logit_splits['Train'] = (train_scaled, train_labels, train_series_per_patch, train_dense)
    if len(val_scaled) > 0:
        logit_splits['Val'] = (val_scaled, val_labels, val_series_per_patch, val_dense)
    if len(test_scaled) > 0:
        logit_splits['Test'] = (test_scaled, test_labels, test_series_per_patch, test_dense)
    if logit_splits:
        config_label = _build_config_label(display_model_name, active_config)
        _save_logit_histograms(
            model, logit_splits, device, plot_dir,
            model_name=display_model_name, config_label=config_label,
        )

    # Logit drift analysis (NoFlow distribution per series)
    drift_stats = {}
    if logit_splits:
        config_label = _build_config_label(display_model_name, active_config)
        drift_stats = _analyse_logit_drift(
            model, logit_splits, device, plot_dir, results_dir,
            model_name=display_model_name, config_label=config_label,
        )

    # ---------------------------------------------------------------
    # Adaptive threshold
    # ---------------------------------------------------------------
    adaptive_cfg = active_config.get('adaptive_threshold', {})
    adaptive_enabled = adaptive_cfg.get('enabled', False)
    adaptive_threshold = 0.5
    adaptive_metrics = {}
    adaptive_per_series = {}

    if adaptive_enabled and len(val_scaled) > 0:
        at_method = adaptive_cfg.get('method', 'gaussian')
        at_percentile = adaptive_cfg.get('percentile', 95.0)
        at_std_mult = adaptive_cfg.get('std_multiplier', 2.0)

        adaptive_threshold = _compute_adaptive_threshold(
            model, val_scaled, val_labels, device,
            method=at_method,
            percentile=at_percentile,
            std_multiplier=at_std_mult,
            val_dense=val_dense,
        )

        # Re-evaluate with adaptive threshold
        if test_loader:
            print(f"\n{'=' * 60}")
            print(f"RE-EVALUATION WITH ADAPTIVE THRESHOLD = {adaptive_threshold:.6g}")
            print(f"{'=' * 60}")
            adaptive_metrics = _evaluate(model, test_loader, device,
                                         threshold=adaptive_threshold)
            _print_metrics(adaptive_metrics)

            adaptive_per_series = _evaluate_per_series(
                model, test_scaled, test_labels, test_series_per_patch, device,
                test_dense=test_dense,
                threshold=adaptive_threshold,
            )
            for sname, smetrics in adaptive_per_series.items():
                print(f"\n{'=' * 60}")
                print(f"ADAPTIVE — {sname}")
                print(f"{'=' * 60}")
                _print_metrics(smetrics)

            # Save adaptive plots
            adaptive_plot_dir = plot_dir / 'adaptive'
            adaptive_plot_dir.mkdir(parents=True, exist_ok=True)
            _save_plots(
                train_losses, val_losses, adaptive_metrics, adaptive_plot_dir,
                model_name=display_model_name, model_config=active_config,
            )
            for sname, smetrics in adaptive_per_series.items():
                _save_plots(
                    train_losses, val_losses, smetrics, adaptive_plot_dir,
                    model_name=display_model_name, model_config=active_config,
                    series_name=sname,
                )

    # Save results
    def _metrics_to_dict(m):
        return {k: float(v) for k, v in m.items()
                if k not in ('y_true', 'y_prob', 'confusion_matrix',
                             'fpr', 'tpr')}

    results = {
        'model_type': model_type,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_losses': [float(v) for v in train_losses],
        'val_losses': [float(v) for v in val_losses],
        'n_params': n_params,
        'pos_weight': float(pos_weight),
        'metrics': _metrics_to_dict(metrics),
    }
    if model_type == 'transformer':
        results['transformer_variant'] = transformer_variant
        results['model_params'] = {
            'd_model': d_model, 'n_heads': n_heads,
            'n_layers': n_layers, 'd_ff': d_ff,
            'dropout': dropout, 'lr': lr, 'batch_size': batch_size,
        }
    if 'confusion_matrix' in metrics:
        results['confusion_matrix'] = metrics['confusion_matrix'].tolist()

    # Add per-series metrics
    if per_series_metrics:
        results['per_series'] = {}
        for sname, smetrics in per_series_metrics.items():
            entry = _metrics_to_dict(smetrics)
            if 'confusion_matrix' in smetrics:
                entry['confusion_matrix'] = smetrics['confusion_matrix'].tolist()
            results['per_series'][sname] = entry

    # Add logit drift statistics
    if drift_stats:
        results['logit_drift'] = drift_stats

    # Add adaptive threshold results
    if adaptive_enabled:
        results['adaptive_threshold'] = {
            'enabled': True,
            'method': adaptive_cfg.get('method', 'gaussian'),
            'threshold': float(adaptive_threshold),
        }
        if adaptive_cfg.get('method', 'gaussian') == 'percentile':
            results['adaptive_threshold']['percentile'] = adaptive_cfg.get('percentile', 95.0)
        else:
            results['adaptive_threshold']['std_multiplier'] = adaptive_cfg.get('std_multiplier', 2.0)

        if adaptive_metrics:
            results['adaptive_metrics'] = _metrics_to_dict(adaptive_metrics)
            if 'confusion_matrix' in adaptive_metrics:
                results['adaptive_confusion_matrix'] = adaptive_metrics['confusion_matrix'].tolist()

        if adaptive_per_series:
            results['adaptive_per_series'] = {}
            for sname, smetrics in adaptive_per_series.items():
                entry = _metrics_to_dict(smetrics)
                if 'confusion_matrix' in smetrics:
                    entry['confusion_matrix'] = smetrics['confusion_matrix'].tolist()
                results['adaptive_per_series'][sname] = entry

    results_json = results_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_json}")
    print(f"{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")

    return results


# ---------------------------------------------------------------------------
# Optuna CNN tuning
# ---------------------------------------------------------------------------

def tune_cnn(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Hyperparameter tuning for CNN classifier using Optuna.

    Optimises CNN architecture and training hyperparameters by maximising
    F1-score on the validation set. The best configuration is saved back
    to mlp.json and a final training run is performed with the best params.

    Search space:
        - cnn_channels: 2-layer combinations from [8, 16, 32, 64]
        - cnn_kernel_size: [3, 5, 7]
        - cnn_fc_dim: [32, 64, 128]
        - dropout: 0.1 - 0.5
        - lr: 1e-4 - 1e-2 (log scale)
        - batch_size: [16, 32, 64]
    """
    import optuna
    from sklearn.metrics import f1_score

    mlp_config = config['classification'].get('mlp', {})
    output_root = mlp_config.get('output_root', './prepared')
    selected_features = mlp_config.get('selected_features', None)
    dense_features = mlp_config.get('dense_features', None) or []
    scaler_type = mlp_config.get('scaler', 'standard')
    epochs = mlp_config.get('epochs', 100)
    patience = mlp_config.get('patience', 10)

    n_trials = mlp_config.get('optuna_n_trials', 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Load data once
    # ------------------------------------------------------------------
    class_base = data_path / output_root
    splits_path = class_base / 'splits.json'
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_dir_name = mlp_config.get('prepared_dir', 'prepared')
    prepared_base = class_base / prepared_dir_name
    with open(prepared_base / 'metadata.json', 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}

    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']

    # Merge CNN + dense features for loading
    all_load_features = list(dict.fromkeys(
        (selected_features or []) + dense_features
    )) or None

    print("=" * 60)
    print("OPTUNA CNN HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Trials:   {n_trials}")
    print(f"Device:   {device}")
    print(f"Epochs:   {epochs}")
    print(f"Patience: {patience}")
    if dense_features:
        print(f"Dense:    {len(dense_features)} features")
    print("=" * 60)

    print("\n--- LOADING DATA ---")
    train_patches, train_labels, _, loaded_feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, all_load_features,
    )
    val_patches, val_labels, _, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, all_load_features,
    )
    print(f"  Train: {train_patches.shape}  Val: {val_patches.shape}")

    # Split CNN / dense features
    loaded_names = loaded_feature_names or []
    if dense_features and loaded_names:
        dense_indices = [i for i, f in enumerate(loaded_names) if f in dense_features]
        cnn_indices = [i for i, f in enumerate(loaded_names) if f not in dense_features]
        n_dense = len(dense_indices)

        def _split_dense_tune(patches):
            if len(patches) == 0:
                return (patches[:, cnn_indices, :] if cnn_indices else patches,
                        np.empty((0, n_dense)))
            return patches[:, cnn_indices, :], patches[:, dense_indices, :].mean(axis=2)

        train_patches, train_dense = _split_dense_tune(train_patches)
        val_patches, val_dense = _split_dense_tune(val_patches)
        print(f"  Dense features: {n_dense}  CNN features: {len(cnn_indices)}")
    else:
        n_dense = 0
        train_dense = val_dense = None

    n_train, n_feat, w = train_patches.shape
    prep_mode = _prep_scaling_mode(metadata_by_id)

    if _skip_train_scaler(prep_mode):
        print(f"  Prep-time scaling active (mode: {prep_mode}) — train-time scaler skipped")
        def _scale(patches_arr):
            return patches_arr
    else:
        # Fit scaler once
        train_flat = train_patches.transpose(0, 2, 1).reshape(-1, n_feat)
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(train_flat)

        def _scale(patches_arr):
            if len(patches_arr) == 0:
                return patches_arr
            n, f, w_ = patches_arr.shape
            flat = patches_arr.transpose(0, 2, 1).reshape(-1, f)
            scaled = scaler.transform(flat)
            return scaled.reshape(n, w_, f).transpose(0, 2, 1)

    # Scale dense separately (still uses standard scaler even under prep scaling)
    if n_dense > 0 and train_dense is not None:
        if scaler_type == 'minmax':
            dense_scaler = MinMaxScaler()
        else:
            dense_scaler = StandardScaler()
        dense_scaler.fit(train_dense)
        train_dense = dense_scaler.transform(train_dense)
        if len(val_dense) > 0:
            val_dense = dense_scaler.transform(val_dense)

    cnn_channel_mode = mlp_config.get('cnn_channel_mode', 'single')
    n_stats = detect_n_stats(loaded_names) if loaded_names else 1

    if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        train_scaled, n_in_channels, h = reshape_to_channels(
            _scale(train_patches), n_feat, w, n_stats)
        val_scaled, _, _ = reshape_to_channels(
            _scale(val_patches), n_feat, w, n_stats)
        print(f"  Dual-channel reshape: ({n_feat},) -> ({n_in_channels}, {h})")
    else:
        n_in_channels = 1
        train_scaled = _scale(train_patches)[:, np.newaxis, :, :]
        val_scaled = _scale(val_patches)[:, np.newaxis, :, :]
        h = n_feat

    # Prepare dense tensors once
    t_train_dense = (torch.tensor(train_dense, dtype=torch.float32)
                     if train_dense is not None and len(train_dense) > 0 else None)
    t_val_dense = (torch.tensor(val_dense, dtype=torch.float32)
                   if val_dense is not None and len(val_dense) > 0 else None)

    def objective(trial):
        # Sample hyperparameters
        ch1 = trial.suggest_categorical('ch1', [8, 16, 32, 64])
        ch2 = trial.suggest_categorical('ch2', [16, 32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        fc_dim = trial.suggest_categorical('fc_dim', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        # Reproducibility per trial
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True

        model = CNNClassifier(
            h, w, channels=[ch1, ch2],
            kernel_size=kernel_size, fc_dim=fc_dim, dropout=dropout,
            in_channels=n_in_channels,
            dense_input_dim=n_dense,
        ).to(device)

        # DataLoaders
        g = torch.Generator().manual_seed(42)
        t_patches = torch.tensor(train_scaled, dtype=torch.float32)
        t_labels = torch.tensor(train_labels, dtype=torch.float32)
        if t_train_dense is not None:
            train_ds = TensorDataset(t_patches, t_train_dense, t_labels)
        else:
            train_ds = TensorDataset(t_patches, t_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(), generator=g,
        )

        v_patches = torch.tensor(val_scaled, dtype=torch.float32)
        v_labels = torch.tensor(val_labels, dtype=torch.float32)
        if t_val_dense is not None:
            val_ds = TensorDataset(v_patches, t_val_dense, v_labels)
        else:
            val_ds = TensorDataset(v_patches, v_labels)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        pos_weight = _compute_pos_weight(train_labels)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )

        def _unpack(batch):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            return batch[0], None, batch[1]

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                patches, dense_batch, labels = _unpack(batch)
                patches = patches.to(device)
                labels = labels.to(device).unsqueeze(1)
                if dense_batch is not None:
                    dense_batch = dense_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(patches, dense=dense_batch), labels)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, dense_batch, labels = _unpack(batch)
                    patches = patches.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    if dense_batch is not None:
                        dense_batch = dense_batch.to(device)
                    val_loss += criterion(model(patches, dense=dense_batch), labels).item()
                    n_val += 1
            avg_val_loss = val_loss / max(n_val, 1)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Compute F1 on validation set with best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in val_loader:
                patches, dense_batch, _ = _unpack(batch)
                patches = patches.to(device)
                if dense_batch is not None:
                    dense_batch = dense_batch.to(device)
                probs = torch.sigmoid(model(patches, dense=dense_batch)).cpu().numpy().flatten()
                all_probs.extend(probs)

        preds = (np.array(all_probs) >= 0.5).astype(int)
        f1 = f1_score(val_labels, preds)

        n_params = sum(p.numel() for p in model.parameters())
        trial.set_user_attr('n_params', n_params)
        trial.set_user_attr('best_val_loss', best_val_loss)

        return f1

    # ------------------------------------------------------------------
    # Run Optuna study
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name='cnn_tune',
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 60}")
    print("BEST TRIAL")
    print(f"{'=' * 60}")
    print(f"  F1 (val):      {best.value:.4f}")
    print(f"  Val loss:      {best.user_attrs['best_val_loss']:.6f}")
    print(f"  Params count:  {best.user_attrs['n_params']:,}")
    print(f"  ch1:           {best.params['ch1']}")
    print(f"  ch2:           {best.params['ch2']}")
    print(f"  kernel_size:   {best.params['kernel_size']}")
    print(f"  fc_dim:        {best.params['fc_dim']}")
    print(f"  dropout:       {best.params['dropout']}")
    print(f"  lr:            {best.params['lr']:.6f}")
    print(f"  batch_size:    {best.params['batch_size']}")

    # ------------------------------------------------------------------
    # Save best params to mlp.json
    # ------------------------------------------------------------------
    mlp_config['cnn_channels'] = [best.params['ch1'], best.params['ch2']]
    mlp_config['cnn_kernel_size'] = best.params['kernel_size']
    mlp_config['cnn_fc_dim'] = best.params['fc_dim']
    mlp_config['dropout'] = best.params['dropout']
    mlp_config['lr'] = round(best.params['lr'], 6)
    mlp_config['batch_size'] = best.params['batch_size']

    # Save best config — try configs/ first, fall back to results/
    import os
    if os.getenv('DOCKER_ENV'):
        config_path = Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'classification' / 'mlp.json'
    else:
        config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'classification' / 'mlp.json'

    results_dir = _scaled_subdir(
        data_path / output_root / 'results', metadata_by_id,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w') as f:
            json.dump(mlp_config, f, indent=2)
        print(f"\n  Config saved: {config_path}")
    except OSError:
        fallback = results_dir / 'best_mlp_config.json'
        with open(fallback, 'w') as f:
            json.dump(mlp_config, f, indent=2)
        print(f"\n  Config dir read-only, saved to: {fallback}")
    study_results = {
        'best_trial': best.number,
        'best_f1': best.value,
        'best_params': best.params,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'value': t.value if t.value is not None else None,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ],
    }
    study_path = results_dir / 'optuna_study.json'
    with open(study_path, 'w') as f:
        json.dump(study_results, f, indent=2)
    print(f"  Study saved:  {study_path}")

    # ------------------------------------------------------------------
    # Final training with best params
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL TRAINING WITH BEST PARAMS")
    print(f"{'=' * 60}")
    return train_model(config, data_path, img_path, plot_enabled,
                       only_files, max_workers)


# ---------------------------------------------------------------------------
# Optuna Transformer tuning
# ---------------------------------------------------------------------------

def tune_transformer(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Hyperparameter tuning for Transformer classifier using Optuna.

    Searches over variant (A/B), d_model, n_heads, n_layers, d_ff,
    dropout, lr, and batch_size. Saves best config to transformer.json
    and runs a final training.
    """
    import optuna
    from sklearn.metrics import f1_score

    trans_config = config['classification'].get('transformer', {})
    output_root = trans_config.get('output_root', './prepared')
    selected_features = trans_config.get('selected_features', None)
    scaler_type = trans_config.get('scaler', 'standard')
    epochs = trans_config.get('epochs', 100)
    patience = trans_config.get('patience', 10)
    n_trials = trans_config.get('optuna_n_trials', 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data once
    class_base = data_path / output_root
    splits_path = class_base / 'splits.json'
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_dir_name = trans_config.get('prepared_dir', 'prepared')
    prepared_base = class_base / prepared_dir_name
    with open(prepared_base / 'metadata.json', 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}

    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']

    print("=" * 60)
    print("OPTUNA TRANSFORMER HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Trials:   {n_trials}")
    print(f"Device:   {device}")
    print(f"Epochs:   {epochs}")
    print(f"Patience: {patience}")
    print("=" * 60)

    print("\n--- LOADING DATA ---")
    train_patches, train_labels, _, loaded_feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, selected_features,
    )
    val_patches, val_labels, _, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, selected_features,
    )
    print(f"  Train: {train_patches.shape}  Val: {val_patches.shape}")

    n_train, n_feat, w = train_patches.shape
    prep_mode = _prep_scaling_mode(metadata_by_id)

    if _skip_train_scaler(prep_mode):
        print(f"  Prep-time scaling active (mode: {prep_mode}) — train-time scaler skipped")
        def _scale(patches_arr):
            return patches_arr
    else:
        # Fit scaler once
        train_flat = train_patches.transpose(0, 2, 1).reshape(-1, n_feat)
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(train_flat)

        def _scale(patches_arr):
            if len(patches_arr) == 0:
                return patches_arr
            n, f, w_ = patches_arr.shape
            flat = patches_arr.transpose(0, 2, 1).reshape(-1, f)
            scaled = scaler.transform(flat)
            return scaled.reshape(n, w_, f).transpose(0, 2, 1)

    train_scaled = _scale(train_patches)
    val_scaled = _scale(val_patches)

    def objective(trial):
        variant = trial.suggest_categorical('variant', ['A', 'B'])

        # d_model must be divisible by n_heads
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        n_heads_options = [h for h in [2, 4, 8] if d_model % h == 0]
        n_heads = trial.suggest_categorical('n_heads', n_heads_options)
        n_layers = trial.suggest_int('n_layers', 1, 4)
        d_ff = trial.suggest_categorical('d_ff', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        # Variant B requires even n_feat
        if variant == 'B' and n_feat % 2 != 0:
            raise optuna.TrialPruned()

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True

        if variant == 'B':
            model = DualStreamTransformer(
                n_feat, w, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ).to(device)
        else:
            model = TransformerClassifier(
                n_feat, w, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ).to(device)

        g = torch.Generator().manual_seed(42)
        t_patches = torch.tensor(train_scaled, dtype=torch.float32)
        t_labels = torch.tensor(train_labels, dtype=torch.float32)
        train_ds = TensorDataset(t_patches, t_labels)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(), generator=g,
        )

        v_patches = torch.tensor(val_scaled, dtype=torch.float32)
        v_labels = torch.tensor(val_labels, dtype=torch.float32)
        val_ds = TensorDataset(v_patches, v_labels)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        pos_weight = _compute_pos_weight(train_labels)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                patches, labels = batch
                patches = patches.to(device)
                labels = labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(patches), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    patches, labels = batch
                    patches = patches.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    val_loss += criterion(model(patches), labels).item()
                    n_val += 1
            avg_val_loss = val_loss / max(n_val, 1)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in val_loader:
                patches, _ = batch
                patches = patches.to(device)
                probs = torch.sigmoid(model(patches)).cpu().numpy().flatten()
                all_probs.extend(probs)

        preds = (np.array(all_probs) >= 0.5).astype(int)
        f1 = f1_score(val_labels, preds)

        n_params = sum(p.numel() for p in model.parameters())
        trial.set_user_attr('n_params', n_params)
        trial.set_user_attr('best_val_loss', best_val_loss)

        return f1

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name='transformer_tune',
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 60}")
    print("BEST TRIAL")
    print(f"{'=' * 60}")
    print(f"  F1 (val):      {best.value:.4f}")
    print(f"  Val loss:      {best.user_attrs['best_val_loss']:.6f}")
    print(f"  Params count:  {best.user_attrs['n_params']:,}")
    for k, v in best.params.items():
        print(f"  {k:<15} {v}")

    # Save best params to transformer.json
    trans_config['transformer_variant'] = best.params['variant']
    trans_config['d_model'] = best.params['d_model']
    trans_config['n_heads'] = best.params['n_heads']
    trans_config['n_layers'] = best.params['n_layers']
    trans_config['d_ff'] = best.params['d_ff']
    trans_config['dropout'] = best.params['dropout']
    trans_config['lr'] = round(best.params['lr'], 6)
    trans_config['batch_size'] = best.params['batch_size']

    # Save best config — try configs/ first, fall back to results/
    import os
    if os.getenv('DOCKER_ENV'):
        config_path = Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'classification' / 'transformer.json'
    else:
        config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'classification' / 'transformer.json'

    results_dir = _scaled_subdir(
        data_path / output_root / 'results', metadata_by_id,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w') as f:
            json.dump(trans_config, f, indent=2)
        print(f"\n  Config saved: {config_path}")
    except OSError:
        fallback = results_dir / 'best_transformer_config.json'
        with open(fallback, 'w') as f:
            json.dump(trans_config, f, indent=2)
        print(f"\n  Config dir read-only, saved to: {fallback}")

    # Save study results (same scaled results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    study_results = {
        'best_trial': best.number,
        'best_f1': best.value,
        'best_params': best.params,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'value': t.value if t.value is not None else None,
                'params': t.params,
                'state': str(t.state),
            }
            for t in study.trials
        ],
    }
    study_path = results_dir / 'optuna_transformer_study.json'
    with open(study_path, 'w') as f:
        json.dump(study_results, f, indent=2)
    print(f"  Study saved:  {study_path}")

    # Final training with best params
    print(f"\n{'=' * 60}")
    print("FINAL TRAINING WITH BEST PARAMS")
    print(f"{'=' * 60}")
    return train_model(config, data_path, img_path, plot_enabled,
                       only_files, max_workers, model_name='transformer')


# ---------------------------------------------------------------------------
# Cross-attention heatmaps (Variant B)
# ---------------------------------------------------------------------------

def _save_attention_heatmaps(
    model: nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    series_per_patch: List[str],
    device: torch.device,
    save_dir: Path,
) -> None:
    """Save cross-attention heatmaps for DualStreamTransformer (Variant B).

    For each test series, collects attention weights and plots the
    average UP->DOWN cross-attention matrix as a heatmap.
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    if not hasattr(model, 'extract_attention_weights'):
        return

    model_dir = save_dir / 'TRANSFORMER_B'
    model_dir.mkdir(parents=True, exist_ok=True)

    series_arr = np.array(series_per_patch)
    unique_series = np.unique(series_arr)

    model.eval()
    batch_size = 64

    for sname in unique_series:
        mask = series_arr == sname
        s_data = test_data[mask]
        if len(s_data) == 0:
            continue

        all_attn_up = []
        with torch.no_grad():
            for i in range(0, len(s_data), batch_size):
                batch = torch.tensor(
                    s_data[i:i + batch_size], dtype=torch.float32,
                ).to(device)
                attn_up, _ = model.extract_attention_weights(batch)
                all_attn_up.append(attn_up.cpu().numpy())

        avg_attn = np.concatenate(all_attn_up, axis=0).mean(axis=0)  # (W, W)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
        ax.set_xlabel('Downstream Time Step')
        ax.set_ylabel('Upstream Time Step')
        ax.set_title(f'Cross-Attention (UP->DOWN) — {sname}')
        fig.colorbar(im, ax=ax)

        safe_name = sname.replace('.', '_').replace(' ', '_')
        fig.savefig(model_dir / f'attention_heatmap_{safe_name}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Attention heatmaps saved to {model_dir}")


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Train XGBoost classifier and evaluate on test set.

    Loads splits.json and prepared patch metadata, loads pre-built patches,
    flattens them to 2D, and trains an XGBClassifier with early stopping.

    Args:
        config: Full config dict (must contain 'classification').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot_enabled: Whether to create plots.
        only_files: Unused.
        max_workers: Unused.

    Returns:
        Dict with training results and metrics.
    """
    import xgboost as xgb

    xgb_config = config['classification'].get('xgboost', {})
    output_root = xgb_config.get('output_root', './prepared')

    n_estimators = xgb_config.get('n_estimators', 300)
    max_depth = xgb_config.get('max_depth', 6)
    learning_rate = xgb_config.get('learning_rate', 0.1)
    subsample = xgb_config.get('subsample', 0.8)
    colsample_bytree = xgb_config.get('colsample_bytree', 0.8)
    scale_pos_weight_cfg = xgb_config.get('scale_pos_weight', 'auto')
    eval_metric = xgb_config.get('eval_metric', 'logloss')
    early_stopping_rounds = xgb_config.get('early_stopping_rounds', 20)

    selected_features = xgb_config.get('selected_features', None)
    dense_features = xgb_config.get('dense_features', None) or []

    print("=" * 60)
    print("TRAIN XGBOOST CLASSIFIER")
    print("=" * 60)
    print(f"n_estimators:       {n_estimators}")
    print(f"max_depth:          {max_depth}")
    print(f"learning_rate:      {learning_rate}")
    print(f"subsample:          {subsample}")
    print(f"colsample_bytree:   {colsample_bytree}")
    print(f"scale_pos_weight:   {scale_pos_weight_cfg}")
    print(f"eval_metric:        {eval_metric}")
    print(f"early_stopping:     {early_stopping_rounds}")
    if selected_features:
        print(f"Selected features:  {len(selected_features)} columns")
    if dense_features:
        print(f"Dense features:     {len(dense_features)} columns")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Load splits.json and prepared metadata
    # ---------------------------------------------------------------
    class_base = data_path / output_root
    splits_path = class_base / 'splits.json'
    if not splits_path.exists():
        print(f"[ERROR] splits.json not found: {splits_path}")
        print("Run 'splitting' step first.")
        return {}

    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_dir_name = xgb_config.get('prepared_dir', 'prepared')
    prepared_base = class_base / prepared_dir_name
    prep_meta_path = prepared_base / 'metadata.json'
    if not prep_meta_path.exists():
        print(f"[ERROR] Prepared metadata not found: {prep_meta_path}")
        print("Run 'prepare' step first.")
        return {}

    with open(prep_meta_path, 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}

    # Results directory — optionally suffixed by prep-time scaling mode
    results_dir = _scaled_subdir(
        data_path / output_root / 'results', metadata_by_id,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    # Merge CNN + dense features for loading
    all_load_features = list(dict.fromkeys(
        (selected_features or []) + dense_features
    )) or None

    # ---------------------------------------------------------------
    # Load pre-built patches
    # ---------------------------------------------------------------
    print("\n--- LOADING TRAINING DATA ---")
    train_patches, train_labels, train_series_per_patch, loaded_feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, all_load_features,
    )

    if len(train_patches) == 0:
        print("[ERROR] No training patches could be loaded.")
        return {}

    print(f"  Train patches: {train_patches.shape}")

    print("\n--- LOADING VALIDATION DATA ---")
    val_patches, val_labels, val_series_per_patch, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, all_load_features,
    )
    print(f"  Val patches: {val_patches.shape}")

    print("\n--- LOADING TEST DATA ---")
    test_patches, test_labels, test_series_per_patch, _ = _load_prepared_data(
        test_ids, prepared_base, metadata_by_id, all_load_features,
    )
    print(f"  Test patches: {test_patches.shape}")

    # ---------------------------------------------------------------
    # Split dense features, average over time, append to flattened
    # ---------------------------------------------------------------
    loaded_names = loaded_feature_names or []
    if dense_features and loaded_names:
        dense_indices = [i for i, f in enumerate(loaded_names) if f in dense_features]
        cnn_indices = [i for i, f in enumerate(loaded_names) if f not in dense_features]
        feature_names = [loaded_names[i] for i in cnn_indices]
        dense_feature_names = [loaded_names[i] for i in dense_indices]
        n_dense = len(dense_indices)
        print(f"\n  CNN features ({len(cnn_indices)}): {feature_names}")
        print(f"  Dense features ({n_dense}): {dense_feature_names}")

        def _split_and_flatten(patches):
            if len(patches) == 0:
                cnn = patches[:, cnn_indices, :] if cnn_indices else patches
                return cnn.reshape(len(cnn), -1), np.empty((0, n_dense))
            cnn = patches[:, cnn_indices, :]
            dense_arr = patches[:, dense_indices, :].mean(axis=2)
            return cnn.reshape(len(cnn), -1), dense_arr

        X_train_cnn, train_dense = _split_and_flatten(train_patches)
        X_val_cnn, val_dense = _split_and_flatten(val_patches)
        X_test_cnn, test_dense = _split_and_flatten(test_patches)

        # Append time-averaged dense columns
        X_train = np.concatenate([X_train_cnn, train_dense], axis=1) if len(X_train_cnn) > 0 else X_train_cnn
        X_val = np.concatenate([X_val_cnn, val_dense], axis=1) if len(X_val_cnn) > 0 else None
        X_test = np.concatenate([X_test_cnn, test_dense], axis=1) if len(X_test_cnn) > 0 else None
        # Feature names for importance: CNN features + dense feature names
        feature_names = feature_names + dense_feature_names
    else:
        feature_names = loaded_names
        n_dense = 0

        X_train = train_patches.reshape(len(train_patches), -1)
        X_val = val_patches.reshape(len(val_patches), -1) if len(val_patches) > 0 else None
        X_test = test_patches.reshape(len(test_patches), -1) if len(test_patches) > 0 else None

    patch_width = train_patches.shape[2]
    y_train = train_labels
    y_val = val_labels if X_val is not None else None
    y_test = test_labels if X_test is not None else None

    print(f"\nFlattened input dim: {X_train.shape[1]}")

    # ---------------------------------------------------------------
    # Compute scale_pos_weight
    # ---------------------------------------------------------------
    if scale_pos_weight_cfg == 'auto':
        pos_weight = _compute_pos_weight(y_train)
    else:
        pos_weight = float(scale_pos_weight_cfg)
    print(f"scale_pos_weight:   {pos_weight:.2f}")

    # ---------------------------------------------------------------
    # Train XGBClassifier
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TRAINING")
    print(f"{'=' * 60}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=pos_weight,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
        use_label_encoder=False,
        verbosity=1,
    )

    eval_set = [(X_train, y_train)]
    if X_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True,
    )

    best_iteration = getattr(model, 'best_iteration', n_estimators)
    print(f"\nBest iteration: {best_iteration}")

    # Save model
    model_path = results_dir / 'xgboost_best.json'
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------
    metrics = {}
    per_series_metrics = {}

    if X_val is not None and len(X_val) > 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON VALIDATION SET")
        print(f"{'=' * 60}")
        y_prob_val = model.predict_proba(X_val)[:, 1]
        val_metrics = _compute_metrics(y_val, y_prob_val)
        _print_metrics(val_metrics)

    if X_test is not None and len(X_test) > 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON TEST SET (overall)")
        print(f"{'=' * 60}")
        y_prob_test = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_prob_test)
        _print_metrics(metrics)

        # Per-series evaluation
        per_series_metrics = _evaluate_per_series_xgboost(
            model, X_test, test_labels, test_series_per_patch,
            already_flat=True,
        )
        for sname, smetrics in per_series_metrics.items():
            print(f"\n{'=' * 60}")
            print(f"EVALUATION ON TEST SET \u2014 {sname}")
            print(f"{'=' * 60}")
            _print_metrics(smetrics)

    elif X_val is not None and len(X_val) > 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON VALIDATION SET (no test data available)")
        print(f"{'=' * 60}")
        y_prob_val = model.predict_proba(X_val)[:, 1]
        metrics = _compute_metrics(y_val, y_prob_val)
        _print_metrics(metrics)

    # ---------------------------------------------------------------
    # Plots (confusion matrix + ROC only, no loss curves for XGBoost)
    # ---------------------------------------------------------------
    # Generate plots — img_path / classification, optionally suffixed by
    # prep-time scaling mode so variants don't overwrite each other.
    plot_dir = _scaled_subdir(img_path / 'classification', metadata_by_id)
    plot_dir.mkdir(parents=True, exist_ok=True)
    _save_plots(
        [], [], metrics, plot_dir,
        model_name="XGBoost", model_config=xgb_config,
    )
    for sname, smetrics in per_series_metrics.items():
        _save_plots(
            [], [], smetrics, plot_dir,
            model_name="XGBoost", model_config=xgb_config,
            series_name=sname,
        )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------
    importance_gain = {}
    importance_perm = {}
    importance_per_series = {}

    if feature_names is not None:
        model_dir = img_path / 'classification' / 'XGBoost'
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("FEATURE IMPORTANCE")
        print(f"{'=' * 60}")

        # 1) Overall gain-based importance
        importance_gain = _aggregate_feature_importance(
            model.feature_importances_, feature_names, patch_width,
            n_dense=n_dense,
        )
        print(f"  Gain-based importance computed ({len(importance_gain)} features)")
        _save_importance_plot(
            importance_gain, model_dir,
            title='Feature Importance (Gain)',
            filename='feature_importance_gain.png',
        )

        # 2) Overall permutation importance on test set
        if X_test is not None and len(X_test) > 0:
            importance_perm = _permutation_importance(
                model, X_test, y_test, feature_names, patch_width,
                n_dense=n_dense,
            )
            print(f"  Permutation importance computed (overall)")
            _save_importance_plot(
                importance_perm, model_dir,
                title='Feature Importance (Permutation)',
                filename='feature_importance_permutation.png',
            )

            # 3) Per-series permutation importance
            series_arr = np.array(test_series_per_patch)
            for sname in np.unique(series_arr):
                mask = series_arr == sname
                if mask.sum() < 2:
                    continue
                s_imp = _permutation_importance(
                    model, X_test[mask], y_test[mask],
                    feature_names, patch_width,
                    n_dense=n_dense,
                )
                importance_per_series[sname] = s_imp
                safe_name = sname.replace('.', '_').replace(' ', '_')
                print(f"  Permutation importance computed ({sname})")
                _save_importance_plot(
                    s_imp, model_dir,
                    title=f'Feature Importance (Permutation) \u2014 {sname}',
                    filename=f'feature_importance_permutation_{safe_name}.png',
                )

            # 4) Comparison plot
            if importance_per_series:
                _save_importance_comparison_plot(
                    importance_perm, importance_per_series, model_dir,
                )
                print(f"  Comparison plot saved")

        print(f"  All importance plots saved to {model_dir}")

    # ---------------------------------------------------------------
    # All-series permutation importance (train + val + test)
    # ---------------------------------------------------------------
    importance_all_series = {}

    analysis_cfg = config.get('analysis', {}).get('feature_importance', {})
    if analysis_cfg.get('compute_all_series', True) and feature_names is not None:
        model_dir = img_path / 'classification' / 'XGBoost'
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("ALL-SERIES FEATURE IMPORTANCE")
        print(f"{'=' * 60}")

        # Combine all data
        all_X_parts = [X_train]
        all_y_parts = [y_train]
        all_series_list = list(train_series_per_patch)

        if X_val is not None and len(X_val) > 0:
            all_X_parts.append(X_val)
            all_y_parts.append(y_val)
            all_series_list.extend(val_series_per_patch)

        if X_test is not None and len(X_test) > 0:
            all_X_parts.append(X_test)
            all_y_parts.append(y_test)
            all_series_list.extend(list(test_series_per_patch))

        all_X = np.concatenate(all_X_parts, axis=0)
        all_y = np.concatenate(all_y_parts, axis=0)
        all_series_arr = np.array(all_series_list)

        n_reps = analysis_cfg.get('n_repeats', 5)
        for sname in np.unique(all_series_arr):
            mask = all_series_arr == sname
            if mask.sum() < 2:
                continue
            s_imp = _permutation_importance(
                model, all_X[mask], all_y[mask],
                feature_names, patch_width, n_repeats=n_reps,
                n_dense=n_dense,
            )
            importance_all_series[sname] = s_imp
            safe_name = sname.replace('.', '_').replace(' ', '_')
            print(f"  Permutation importance computed ({sname})")
            _save_importance_plot(
                s_imp, model_dir,
                title=f'Feature Importance (Permutation) \u2014 {sname} (all data)',
                filename=f'feature_importance_all_{safe_name}.png',
            )

        # Stability plot
        if importance_all_series:
            top_n = analysis_cfg.get('top_n', 10)
            _save_importance_stability_plot(
                importance_all_series, model_dir, top_n=top_n,
            )
            print(f"  Stability plot saved")

        print(f"  All-series importance plots saved to {model_dir}")

    # ---------------------------------------------------------------
    # Adaptive threshold
    # ---------------------------------------------------------------
    adaptive_cfg = xgb_config.get('adaptive_threshold', {})
    adaptive_enabled = adaptive_cfg.get('enabled', False)
    adaptive_threshold = 0.5
    adaptive_metrics = {}
    adaptive_per_series = {}

    if adaptive_enabled and X_val is not None and len(X_val) > 0:
        at_method = adaptive_cfg.get('method', 'gaussian')
        at_percentile = adaptive_cfg.get('percentile', 95.0)
        at_std_mult = adaptive_cfg.get('std_multiplier', 2.0)

        # Get NoFlow probs from validation set
        val_probs = model.predict_proba(X_val)[:, 1]
        noflow_probs = val_probs[y_val == 0]

        adaptive_threshold = _compute_adaptive_threshold_from_probs(
            noflow_probs,
            method=at_method,
            percentile=at_percentile,
            std_multiplier=at_std_mult,
        )

        # Re-evaluate with adaptive threshold
        if X_test is not None and len(X_test) > 0:
            print(f"\n{'=' * 60}")
            print(f"RE-EVALUATION WITH ADAPTIVE THRESHOLD = {adaptive_threshold:.6g}")
            print(f"{'=' * 60}")
            y_prob_test_adaptive = model.predict_proba(X_test)[:, 1]
            adaptive_metrics = _compute_metrics(y_test, y_prob_test_adaptive,
                                                threshold=adaptive_threshold)
            _print_metrics(adaptive_metrics)

            adaptive_per_series = _evaluate_per_series_xgboost(
                model, X_test, test_labels, test_series_per_patch,
                already_flat=True,
                threshold=adaptive_threshold,
            )
            for sname, smetrics in adaptive_per_series.items():
                print(f"\n{'=' * 60}")
                print(f"ADAPTIVE — {sname}")
                print(f"{'=' * 60}")
                _print_metrics(smetrics)

    # ---------------------------------------------------------------
    # Save results JSON
    # ---------------------------------------------------------------
    def _metrics_to_dict_xgb(m):
        return {k: float(v) for k, v in m.items()
                if k not in ('y_true', 'y_prob', 'confusion_matrix',
                             'fpr', 'tpr')}

    results = {
        'best_iteration': best_iteration,
        'scale_pos_weight': float(pos_weight),
        'metrics': _metrics_to_dict_xgb(metrics),
    }
    if 'confusion_matrix' in metrics:
        results['confusion_matrix'] = metrics['confusion_matrix'].tolist()

    if per_series_metrics:
        results['per_series'] = {}
        for sname, smetrics in per_series_metrics.items():
            entry = _metrics_to_dict_xgb(smetrics)
            if 'confusion_matrix' in smetrics:
                entry['confusion_matrix'] = smetrics['confusion_matrix'].tolist()
            results['per_series'][sname] = entry

    if importance_gain:
        results['feature_importance_gain'] = importance_gain
    if importance_perm:
        results['feature_importance_permutation'] = importance_perm
    if importance_per_series:
        results['feature_importance_per_series'] = importance_per_series
    if importance_all_series:
        results['feature_importance_all_series'] = importance_all_series

    # Add adaptive threshold results
    if adaptive_enabled:
        results['adaptive_threshold'] = {
            'enabled': True,
            'method': adaptive_cfg.get('method', 'gaussian'),
            'threshold': float(adaptive_threshold),
        }
        if adaptive_metrics:
            results['adaptive_metrics'] = _metrics_to_dict_xgb(adaptive_metrics)
            if 'confusion_matrix' in adaptive_metrics:
                results['adaptive_confusion_matrix'] = adaptive_metrics['confusion_matrix'].tolist()
        if adaptive_per_series:
            results['adaptive_per_series'] = {}
            for sname, smetrics in adaptive_per_series.items():
                entry = _metrics_to_dict_xgb(smetrics)
                if 'confusion_matrix' in smetrics:
                    entry['confusion_matrix'] = smetrics['confusion_matrix'].tolist()
                results['adaptive_per_series'][sname] = entry

    results_json = results_dir / 'results_xgboost.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_json}")
    print(f"{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")

    return results


# ---------------------------------------------------------------------------
# Latent feature extraction + visualization
# ---------------------------------------------------------------------------

def extract_latent(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Extract CNN latent features and visualize with t-SNE + GMM clustering.

    Loads a trained CNN model, extracts the feature vectors after the
    adaptive pooling layer (before classifier), and produces:
    - t-SNE scatter colored by label (Flow/NoFlow)
    - t-SNE scatter colored by series
    - t-SNE scatter colored by GMM cluster assignment
    - GMM cluster purity analysis

    Args:
        config: Full config dict (must contain 'classification').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot_enabled: Unused (always produces plots).
        only_files: Unused.
        max_workers: Unused.

    Returns:
        Dict with cluster analysis results.
    """
    from sklearn.manifold import TSNE
    from sklearn.mixture import GaussianMixture

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    mlp_config = config['classification'].get('mlp', {})
    trans_config = config['classification'].get('transformer', {})
    output_root = mlp_config.get('output_root', './prepared')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------------
    # Pre-load metadata to determine scaling-mode subdirectory
    # (needed before results.json lookup so we find the right variant)
    # ---------------------------------------------------------------
    class_base_early = data_path / output_root
    prep_dir_early = mlp_config.get('prepared_dir', 'prepared')
    prep_meta_early = class_base_early / prep_dir_early / 'metadata.json'
    metadata_by_id: Dict[int, dict] = {}
    if prep_meta_early.exists():
        with open(prep_meta_early, 'r') as f:
            metadata_by_id = {m['segment_id']: m for m in json.load(f)}

    # ---------------------------------------------------------------
    # Detect saved model type from results.json (within scaled subdir)
    # ---------------------------------------------------------------
    results_dir = _scaled_subdir(
        data_path / output_root / 'results', metadata_by_id,
    )
    results_json_path = results_dir / 'results.json'
    saved_model_type = 'cnn'  # default for backward compat
    saved_transformer_variant = 'A'
    if results_json_path.exists():
        with open(results_json_path, 'r') as f:
            saved_results = json.load(f)
        saved_model_type = saved_results.get('model_type', 'cnn')
        saved_transformer_variant = saved_results.get('transformer_variant', 'A')

    # Use config from the matching model
    if saved_model_type == 'transformer':
        active_config = trans_config
        selected_features = trans_config.get('selected_features', None)
        dense_features = []
        scaler_type = trans_config.get('scaler', 'standard')
        dropout = trans_config.get('dropout', 0.3)
        cnn_channel_mode = 'single'  # not used
    else:
        active_config = mlp_config
        selected_features = mlp_config.get('selected_features', None)
        dense_features = mlp_config.get('dense_features', None) or []
        scaler_type = mlp_config.get('scaler', 'standard')
        dropout = mlp_config.get('dropout', 0.3)
        cnn_channel_mode = mlp_config.get('cnn_channel_mode', 'single')

    # CNN params (needed to reconstruct CNN model)
    cnn_channels = mlp_config.get('cnn_channels', [16, 32])
    cnn_kernel_size = mlp_config.get('cnn_kernel_size', 5)
    cnn_fc_dim = mlp_config.get('cnn_fc_dim', 64)

    print("=" * 60)
    print("LATENT FEATURE EXTRACTION & VISUALIZATION")
    print("=" * 60)
    print(f"Device:            {device}")
    print(f"Saved model type:  {saved_model_type}")
    if saved_model_type == 'transformer':
        print(f"Variant:           {saved_transformer_variant}")
    if selected_features:
        print(f"Selected features: {len(selected_features)} columns")
    if dense_features:
        print(f"Dense features:    {len(dense_features)} columns")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Load splits + metadata
    # ---------------------------------------------------------------
    class_base = data_path / output_root
    splits_path = class_base / 'splits.json'
    if not splits_path.exists():
        print(f"[ERROR] splits.json not found: {splits_path}")
        return {}

    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_dir_name = active_config.get('prepared_dir', 'prepared')
    prepared_base = class_base / prepared_dir_name
    with open(prepared_base / 'metadata.json', 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}

    # Merge features for loading
    all_features = list(dict.fromkeys(
        (selected_features or []) + dense_features
    )) or None

    # ---------------------------------------------------------------
    # Load all splits
    # ---------------------------------------------------------------
    all_patches_list = []
    all_labels_list = []
    all_series_list = []
    all_splits_list = []

    for split_name, split_key in [('Train', 'train'), ('Val', 'val'), ('Test', 'test')]:
        seg_ids = splits_data['splits'].get(split_key, [])
        if not seg_ids:
            continue
        patches, labels, series, _ = _load_prepared_data(
            seg_ids, prepared_base, metadata_by_id, all_features,
        )
        if len(patches) == 0:
            continue
        all_patches_list.append(patches)
        all_labels_list.append(labels)
        all_series_list.extend(series)
        all_splits_list.extend([split_name] * len(labels))
        print(f"  {split_name}: {patches.shape[0]} patches")

    if not all_patches_list:
        print("[ERROR] No data loaded.")
        return {}

    all_patches = np.concatenate(all_patches_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    all_series = np.array(all_series_list)
    all_splits = np.array(all_splits_list)

    # ---------------------------------------------------------------
    # Split CNN / dense features
    # ---------------------------------------------------------------
    loaded_names = None
    # Re-derive feature names from first segment
    for m in all_prep_metadata:
        if 'feature_names' in m:
            names = m['feature_names']
            if all_features:
                loaded_names = [n for n in names if n in all_features]
            else:
                loaded_names = names
            break

    if dense_features and loaded_names:
        dense_indices = [i for i, f in enumerate(loaded_names) if f in dense_features]
        cnn_indices = [i for i, f in enumerate(loaded_names) if f not in dense_features]
        n_dense = len(dense_indices)
        cnn_patches = all_patches[:, cnn_indices, :]
        dense_arr = all_patches[:, dense_indices, :].mean(axis=2)
    else:
        n_dense = 0
        cnn_patches = all_patches
        dense_arr = None

    # ---------------------------------------------------------------
    # Scale
    # ---------------------------------------------------------------
    n_total, n_feat, w = cnn_patches.shape
    train_mask = all_splits == 'Train'
    prep_mode = _prep_scaling_mode(metadata_by_id)

    if _skip_train_scaler(prep_mode):
        print(f"  Prep-time scaling active (mode: {prep_mode}) — train-time scaler skipped")
        def _scale(patches_arr):
            return patches_arr
    else:
        train_flat = cnn_patches[train_mask].transpose(0, 2, 1).reshape(-1, n_feat)
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(train_flat)

        def _scale(patches_arr):
            if len(patches_arr) == 0:
                return patches_arr
            n, f, w_ = patches_arr.shape
            flat = patches_arr.transpose(0, 2, 1).reshape(-1, f)
            scaled = scaler.transform(flat)
            return scaled.reshape(n, w_, f).transpose(0, 2, 1)

    scaled = _scale(cnn_patches)

    if n_dense > 0 and dense_arr is not None:
        if scaler_type == 'minmax':
            dense_scaler = MinMaxScaler()
        else:
            dense_scaler = StandardScaler()
        dense_scaler.fit(dense_arr[train_mask])
        dense_arr = dense_scaler.transform(dense_arr)

    # Reshape for model input
    n_stats = detect_n_stats(loaded_names) if loaded_names else 1
    if saved_model_type == 'transformer':
        # Transformer takes (B, F, W) directly — no channel dim
        h = n_feat
        n_in_channels = 1
        print(f"  Transformer input: ({n_feat}, {w}) — no channel reshape")
    elif cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        scaled, n_in_channels, h = reshape_to_channels(
            scaled, n_feat, w, n_stats)
        print(f"  Dual-channel reshape: ({n_feat},) -> ({n_in_channels}, {h})")
    else:
        n_in_channels = 1
        scaled = scaled[:, np.newaxis, :, :]
        h = n_feat

    # ---------------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------------
    model_path = results_dir / 'mlp_best.pt'
    if not model_path.exists():
        print(f"[ERROR] Trained model not found: {model_path}")
        print("Run 'train' step first.")
        return {}

    if saved_model_type == 'transformer':
        t_cfg = trans_config
        t_d_model = t_cfg.get('d_model', 64)
        t_n_heads = t_cfg.get('n_heads', 4)
        t_n_layers = t_cfg.get('n_layers', 2)
        t_d_ff = t_cfg.get('d_ff', 128)
        if saved_transformer_variant == 'B':
            model = DualStreamTransformer(
                n_feat, w, d_model=t_d_model, n_heads=t_n_heads,
                n_layers=t_n_layers, d_ff=t_d_ff, dropout=dropout,
            )
        else:
            model = TransformerClassifier(
                n_feat, w, d_model=t_d_model, n_heads=t_n_heads,
                n_layers=t_n_layers, d_ff=t_d_ff, dropout=dropout,
            )
    else:
        model = CNNClassifier(
            h, w, channels=cnn_channels,
            kernel_size=cnn_kernel_size, fc_dim=cnn_fc_dim,
            dropout=dropout, in_channels=n_in_channels,
            dense_input_dim=n_dense,
        )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"\nModel loaded: {model_path}")

    # ---------------------------------------------------------------
    # Extract features
    # ---------------------------------------------------------------
    print("\n--- EXTRACTING LATENT FEATURES ---")
    batch_size = 64
    all_feats = []
    t_scaled = torch.tensor(scaled, dtype=torch.float32)
    t_dense = (torch.tensor(dense_arr, dtype=torch.float32)
               if dense_arr is not None else None)

    with torch.no_grad():
        for i in range(0, len(t_scaled), batch_size):
            batch = t_scaled[i:i + batch_size].to(device)
            d_batch = None
            if t_dense is not None:
                d_batch = t_dense[i:i + batch_size].to(device)
            feats = model.extract_features(batch, dense=d_batch)
            all_feats.append(feats.cpu().numpy())

    latent = np.concatenate(all_feats, axis=0)
    print(f"  Latent shape: {latent.shape}  ({latent.shape[1]} dims)")

    # ---------------------------------------------------------------
    # t-SNE
    # ---------------------------------------------------------------
    print("\n--- RUNNING t-SNE ---")
    perplexity = min(30, len(latent) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(latent)
    print(f"  t-SNE done. Shape: {embedding.shape}")

    # ---------------------------------------------------------------
    # GMM clustering
    # ---------------------------------------------------------------
    print("\n--- GMM CLUSTERING ---")
    n_clusters = len(np.unique(all_series)) + 1  # series count + 1
    n_clusters = min(max(n_clusters, 2), 10)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    clusters = gmm.fit_predict(latent)
    print(f"  GMM with {n_clusters} components")

    # Cluster purity analysis
    print(f"\n  {'Cluster':<10} {'Size':>6} {'Flow%':>7} {'NoFlow%':>8}  {'Dominant Series'}")
    print(f"  {'-' * 55}")
    cluster_info = {}
    for c in range(n_clusters):
        mask = clusters == c
        size = mask.sum()
        if size == 0:
            continue
        flow_pct = (all_labels[mask] == 1).mean() * 100
        noflow_pct = (all_labels[mask] == 0).mean() * 100
        # Find dominant series
        series_in_cluster = all_series[mask]
        unique_s, counts = np.unique(series_in_cluster, return_counts=True)
        dominant = unique_s[counts.argmax()]
        dominant_pct = counts.max() / size * 100
        print(f"  {c:<10} {size:>6} {flow_pct:>6.1f}% {noflow_pct:>7.1f}%  "
              f"{dominant} ({dominant_pct:.0f}%)")
        cluster_info[int(c)] = {
            'size': int(size),
            'flow_pct': round(flow_pct, 1),
            'noflow_pct': round(noflow_pct, 1),
            'dominant_series': dominant,
            'dominant_pct': round(dominant_pct, 1),
        }

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    plot_dir = _scaled_subdir(
        img_path / 'classification' / 'latent', metadata_by_id,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    def _sort_series_key(name):
        for fmt in ('%d.%m.%Y', '%d.%m.%y'):
            try:
                return datetime.strptime(name, fmt)
            except ValueError:
                continue
        return datetime.max

    unique_series = sorted(np.unique(all_series), key=_sort_series_key)
    series_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_series), 1)))
    series_cmap = {s: series_colors[i] for i, s in enumerate(unique_series)}

    # --- Plot 1: colored by label ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for label_val, label_name, color in [(0, 'NoFlow', 'tab:blue'), (1, 'Flow', 'tab:red')]:
        mask = all_labels == label_val
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, label=label_name, alpha=0.4, s=8, edgecolors='none')
    ax.set_title('CNN Latent Space — by Label')
    ax.legend(markerscale=3)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.savefig(plot_dir / 'tsne_by_label.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: colored by series ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for sname in unique_series:
        mask = all_series == sname
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[series_cmap[sname]], label=sname, alpha=0.4, s=8, edgecolors='none')
    ax.set_title('CNN Latent Space — by Series')
    ax.legend(markerscale=3)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.savefig(plot_dir / 'tsne_by_series.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 3: colored by series + label (shape encodes label) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for sname in unique_series:
        for label_val, marker in [(0, 'o'), (1, '^')]:
            mask = (all_series == sname) & (all_labels == label_val)
            if not mask.any():
                continue
            label_str = f"{sname} {'Flow' if label_val else 'NoFlow'}"
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[series_cmap[sname]], marker=marker,
                       label=label_str, alpha=0.4, s=12, edgecolors='none')
    ax.set_title('CNN Latent Space — by Series & Label')
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.savefig(plot_dir / 'tsne_by_series_label.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 4: colored by GMM cluster ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    for c in range(n_clusters):
        mask = clusters == c
        if not mask.any():
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cluster_colors[c]], label=f'Cluster {c}',
                   alpha=0.4, s=8, edgecolors='none')
    ax.set_title(f'CNN Latent Space — GMM ({n_clusters} clusters)')
    ax.legend(markerscale=3)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.savefig(plot_dir / 'tsne_by_cluster.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 5: per-series subplots (label colored) ---
    n_series = len(unique_series)
    cols = min(3, n_series)
    rows = (n_series + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, sname in enumerate(unique_series):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        # Background: all points gray
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c='lightgray', alpha=0.1, s=4, edgecolors='none')
        # Foreground: this series
        for label_val, label_name, color in [(0, 'NoFlow', 'tab:blue'), (1, 'Flow', 'tab:red')]:
            mask = (all_series == sname) & (all_labels == label_val)
            if mask.any():
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, label=f'{label_name} ({mask.sum()})',
                           alpha=0.5, s=10, edgecolors='none')
        ax.set_title(sname, fontweight='bold')
        ax.legend(fontsize=7, markerscale=2)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    # Hide unused
    for idx in range(n_series, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    fig.suptitle('CNN Latent Space — per Series', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plot_dir / 'tsne_per_series.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  All plots saved to {plot_dir}")

    # Save results
    results = {
        'n_patches': int(len(latent)),
        'latent_dim': int(latent.shape[1]),
        'n_clusters': n_clusters,
        'cluster_info': cluster_info,
    }
    results_path = results_dir / 'latent_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    print(f"\n{'=' * 60}")
    print("LATENT ANALYSIS COMPLETE")
    print(f"{'=' * 60}")

    return results


def _evaluate_per_series_xgboost(
    model,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    series_per_patch: List[str],
    already_flat: bool = False,
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """Evaluate XGBoost model separately for each test series."""
    if len(test_data) == 0 or not series_per_patch:
        return {}

    series_arr = np.array(series_per_patch)
    unique_series = np.unique(series_arr)

    results = {}
    for sname in unique_series:
        mask = series_arr == sname
        if not mask.any():
            continue

        s_data = test_data[mask]
        if not already_flat:
            s_data = s_data.reshape(mask.sum(), -1)
        s_labels = test_labels[mask]

        y_prob = model.predict_proba(s_data)[:, 1]
        results[sname] = _compute_metrics(s_labels, y_prob, threshold=threshold)

    return results


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute classification metrics from ground-truth labels and probabilities.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to predictions.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, y_true, y_prob, fpr, tpr, threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        roc_auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        roc_auc = 0.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': np.array([[tn, fn], [fp, tp]]),
        'y_true': y_true,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr,
        'threshold': threshold,
    }


def _evaluate_per_series(
    model: nn.Module,
    test_patches: np.ndarray,
    test_labels: np.ndarray,
    series_per_patch: List[str],
    device: torch.device,
    test_dense: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """
    Evaluate the model separately for each test series.

    Uses in-memory data with series tracking per patch.

    Returns:
        Dict mapping series name to metrics dict.
    """
    if len(test_patches) == 0 or not series_per_patch:
        return {}

    series_arr = np.array(series_per_patch)
    unique_series = np.unique(series_arr)

    results = {}
    for sname in unique_series:
        mask = series_arr == sname
        if not mask.any():
            continue

        s_patches = torch.tensor(test_patches[mask], dtype=torch.float32)
        s_labels = test_labels[mask]
        s_dense = None
        if test_dense is not None and len(test_dense) > 0:
            s_dense = torch.tensor(test_dense[mask], dtype=torch.float32)

        model.eval()
        all_probs = []
        infer_batch = 64
        with torch.no_grad():
            for i in range(0, len(s_patches), infer_batch):
                batch = s_patches[i:i + infer_batch].to(device)
                dense_batch = None
                if s_dense is not None:
                    dense_batch = s_dense[i:i + infer_batch].to(device)
                logits = model(batch, dense=dense_batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)

        results[sname] = _compute_metrics(s_labels, np.array(all_probs),
                                          threshold=threshold)

    return results


def _evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Run evaluation and compute all metrics."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                patches, dense_batch, labels = batch
                dense_batch = dense_batch.to(device)
            else:
                patches, labels = batch
                dense_batch = None
            patches = patches.to(device)
            logits = model(patches, dense=dense_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    return _compute_metrics(np.array(all_labels), np.array(all_probs),
                            threshold=threshold)


def _print_metrics(metrics: Dict) -> None:
    """Print evaluation metrics in a formatted table."""
    threshold = metrics.get('threshold', 0.5)
    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 32}")
    if threshold < 1e-3:
        print(f"  {'Threshold':<20} {threshold:>10.2e}")
    else:
        print(f"  {'Threshold':<20} {threshold:>10.4f}")
    print(f"  {'Accuracy':<20} {metrics['accuracy']:>10.4f}")
    print(f"  {'Precision':<20} {metrics['precision']:>10.4f}")
    print(f"  {'Recall':<20} {metrics['recall']:>10.4f}")
    print(f"  {'F1-Score':<20} {metrics['f1']:>10.4f}")
    print(f"  {'ROC-AUC':<20} {metrics['roc_auc']:>10.4f}")

    cm = metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"                      Actual")
    print(f"                    NoLeak  Leak")
    print(f"  Pred NoLeak    [{cm[0, 0]:>6d}  {cm[0, 1]:>6d}]")
    print(f"  Pred Leak      [{cm[1, 0]:>6d}  {cm[1, 1]:>6d}]")


def _build_config_label(model_name: str, model_config: dict) -> str:
    """Build a short label string from model config for filenames and titles."""
    parts = [model_name]
    if 'hidden_dims' in model_config:
        dims = 'x'.join(str(d) for d in model_config['hidden_dims'])
        parts.append(f"h{dims}")
    if 'dropout' in model_config:
        parts.append(f"d{model_config['dropout']}")
    if 'lr' in model_config:
        parts.append(f"lr{model_config['lr']}")
    if 'batch_size' in model_config:
        parts.append(f"bs{model_config['batch_size']}")
    return "_".join(parts)


def _collect_logits(
    model: nn.Module,
    patches: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    dense: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run forward pass and return raw logits (before sigmoid)."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.tensor(
                patches[i:i + batch_size], dtype=torch.float32,
            ).to(device)
            dense_batch = None
            if dense is not None and len(dense) > 0:
                dense_batch = torch.tensor(
                    dense[i:i + batch_size], dtype=torch.float32,
                ).to(device)
            logits = model(batch, dense=dense_batch).cpu().numpy().flatten()
            all_logits.extend(logits)
    return np.array(all_logits)


def _save_logit_histograms(
    model: nn.Module,
    splits_data: Dict[str, tuple],
    device: torch.device,
    save_dir: Path,
    model_name: str = "CNN",
    config_label: str = "",
) -> None:
    """Save per-split and per-series logit distribution histograms.

    Layout:
        Row 0: One overall subplot per split (Train, Val, Test).
        Rows 1+: One subplot per unique series (chronological), with the
        split indicated in the title. Arranged in a 3-column grid.

    Args:
        splits_data: Dict mapping split name (e.g. 'Train', 'Val', 'Test')
            to (scaled_patches, labels, series_per_patch) or
            (scaled_patches, labels, series_per_patch, dense_arr).
        save_dir: Directory for saving plots.
        model_name: Model name for subfolder and titles.
        config_label: Config label string for filenames.
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from datetime import datetime

    model_dir = save_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Collect logits for all splits and series
    overall_logits: Dict[str, Dict[str, np.ndarray]] = {}
    # series_entries: list of (series_name, split_name, {noflow: arr, flow: arr})
    series_entries: List[Tuple[str, str, Dict[str, np.ndarray]]] = []

    for split_name, split_tuple in splits_data.items():
        patches, labels, series_per_patch = split_tuple[0], split_tuple[1], split_tuple[2]
        dense_arr = split_tuple[3] if len(split_tuple) > 3 else None
        if len(patches) == 0:
            continue
        logits = _collect_logits(model, patches, labels, device, dense=dense_arr)
        series_arr = np.array(series_per_patch)

        overall_logits[split_name] = {
            'noflow': logits[labels == 0],
            'flow': logits[labels == 1],
        }

        for sname in sorted(np.unique(series_arr)):
            mask = series_arr == sname
            series_entries.append((sname, split_name, {
                'noflow': logits[mask & (labels == 0)],
                'flow': logits[mask & (labels == 1)],
            }))

    if not overall_logits:
        return

    # Sort series entries chronologically by date in series name
    def _sort_key(entry):
        sname = entry[0]
        # Parse dates like "10.01.2026" or "7.02.2026"
        for fmt in ('%d.%m.%Y', '%d.%m.%y'):
            try:
                return datetime.strptime(sname, fmt)
            except ValueError:
                continue
        return datetime.max

    series_entries.sort(key=_sort_key)

    # Split name display order
    split_order = [s for s in ['Train', 'Val', 'Test'] if s in overall_logits]

    # Global logit range for consistent x-axis
    all_logit_vals = []
    for entry in overall_logits.values():
        for arr in entry.values():
            if len(arr) > 0:
                all_logit_vals.append(arr)
    if all_logit_vals:
        all_concat = np.concatenate(all_logit_vals)
        x_min, x_max = np.percentile(all_concat, [1, 99])
        margin = (x_max - x_min) * 0.1
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = -5, 5

    bins = np.linspace(x_min, x_max, 50)

    # Layout: row 0 = overalls (up to 3 cols), then series in 3-col grid
    grid_cols = 3
    n_overalls = len(split_order)
    n_series = len(series_entries)
    series_rows = (n_series + grid_cols - 1) // grid_cols
    n_rows = 1 + series_rows  # 1 overall row + series rows

    fig, axes = plt.subplots(
        n_rows, grid_cols,
        figsize=(5 * grid_cols, 3.2 * n_rows),
        squeeze=False,
    )

    # Mapping split -> color accent for title background
    split_colors = {'Train': '#d4edda', 'Val': '#fff3cd', 'Test': '#d1ecf1'}

    def _plot_hist(ax, data_dict, title, show_threshold_label=False):
        nf = data_dict['noflow']
        fl = data_dict['flow']
        if len(nf) > 0:
            ax.hist(nf, bins=bins, histtype='step', linewidth=1.5,
                    label=f'NoFlow (n={len(nf)})', color='tab:blue')
        if len(fl) > 0:
            ax.hist(fl, bins=bins, histtype='step', linewidth=1.5,
                    label=f'Flow (n={len(fl)})', color='tab:red')
        kw = dict(label='threshold=0') if show_threshold_label else {}
        ax.axvline(0, color='grey', linestyle='--', alpha=0.5, **kw)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel('Count')

    # Row 0: overall per split
    for col, split_name in enumerate(split_order):
        _plot_hist(
            axes[0, col], overall_logits[split_name],
            f'{split_name} (all series)', show_threshold_label=True,
        )
        axes[0, col].set_facecolor(split_colors.get(split_name, 'white'))

    # Hide unused cells in overall row
    for col in range(n_overalls, grid_cols):
        axes[0, col].set_visible(False)

    # Rows 1+: per-series chronologically in 3-col grid
    for idx, (sname, split_name, data_dict) in enumerate(series_entries):
        row = 1 + idx // grid_cols
        col = idx % grid_cols
        _plot_hist(
            axes[row, col], data_dict,
            f'{sname}  ({split_name})',
        )
        axes[row, col].set_facecolor(split_colors.get(split_name, 'white'))

    # Hide unused cells in last series row
    n_used_last_row = n_series % grid_cols
    if n_used_last_row > 0:
        for col in range(n_used_last_row, grid_cols):
            axes[n_rows - 1, col].set_visible(False)

    # x-label on visible bottom cells
    for col in range(grid_cols):
        for row in range(n_rows - 1, -1, -1):
            if axes[row, col].get_visible():
                axes[row, col].set_xlabel('Logit value')
                break

    fig.suptitle(
        f'Logit Distributions per Split & Series ({config_label})',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    fname = model_dir / f'logit_distributions_{config_label}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Logit distributions saved to {fname}")


def _compute_adaptive_threshold(
    model: nn.Module,
    val_patches: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    method: str = "gaussian",
    percentile: float = 95.0,
    std_multiplier: float = 2.0,
    val_dense: Optional[np.ndarray] = None,
) -> float:
    """Compute adaptive decision threshold from validation NoFlow logits.

    Collects raw logits for all validation NoFlow patches, fits a
    distribution, and returns a probability threshold (after sigmoid).

    Args:
        model: Trained model.
        val_patches: Validation patches (scaled).
        val_labels: Validation labels (0/1).
        device: Torch device.
        method: Threshold method — "gaussian" (mean + k*std) or
                "percentile" (Nth percentile of NoFlow logits).
        percentile: Percentile to use when method="percentile".
        std_multiplier: Number of standard deviations above NoFlow mean
                        when method="gaussian".
        val_dense: Optional dense features for validation patches.

    Returns:
        Probability threshold in [0, 1].
    """
    noflow_mask = val_labels == 0
    nf_patches = val_patches[noflow_mask]

    if len(nf_patches) == 0:
        print("  [WARN] No NoFlow patches in validation set — using default threshold 0.5")
        return 0.5

    nf_dense = None
    if val_dense is not None and len(val_dense) > 0:
        nf_dense = val_dense[noflow_mask]

    nf_logits = _collect_logits(model, nf_patches, val_labels[noflow_mask],
                                device, dense=nf_dense)

    if method == "percentile":
        logit_threshold = float(np.percentile(nf_logits, percentile))
    else:
        # Gaussian: mean + k * std
        nf_mean = float(np.mean(nf_logits))
        nf_std = float(np.std(nf_logits))
        logit_threshold = nf_mean + std_multiplier * nf_std

    prob_threshold = float(1.0 / (1.0 + np.exp(-logit_threshold)))

    print(f"\n  Adaptive threshold ({method}):")
    print(f"    NoFlow logits: mean={np.mean(nf_logits):.3f}, "
          f"std={np.std(nf_logits):.3f}, n={len(nf_logits)}")
    if method == "percentile":
        print(f"    Percentile: {percentile}th → logit={logit_threshold:.3f}")
    else:
        print(f"    mean + {std_multiplier}*std → logit={logit_threshold:.3f}")
    print(f"    Probability threshold: {prob_threshold:.6g}")

    return prob_threshold


def _compute_adaptive_threshold_from_probs(
    noflow_probs: np.ndarray,
    method: str = "gaussian",
    percentile: float = 95.0,
    std_multiplier: float = 2.0,
) -> float:
    """Compute adaptive threshold from NoFlow probabilities (for XGBoost).

    Args:
        noflow_probs: Predicted probabilities for NoFlow patches.
        method: "gaussian" or "percentile".
        percentile: Percentile when method="percentile".
        std_multiplier: Std multiplier when method="gaussian".

    Returns:
        Probability threshold in [0, 1].
    """
    if len(noflow_probs) == 0:
        return 0.5

    if method == "percentile":
        threshold = float(np.percentile(noflow_probs, percentile))
    else:
        nf_mean = float(np.mean(noflow_probs))
        nf_std = float(np.std(noflow_probs))
        threshold = nf_mean + std_multiplier * nf_std

    threshold = float(np.clip(threshold, 0.01, 0.99))

    print(f"\n  Adaptive threshold ({method}):")
    print(f"    NoFlow probs: mean={np.mean(noflow_probs):.4f}, "
          f"std={np.std(noflow_probs):.4f}, n={len(noflow_probs)}")
    if method == "percentile":
        print(f"    Percentile: {percentile}th → threshold={threshold:.4f}")
    else:
        print(f"    mean + {std_multiplier}*std → threshold={threshold:.4f}")

    return threshold


def _analyse_logit_drift(
    model: nn.Module,
    splits_data: Dict[str, tuple],
    device: torch.device,
    save_dir: Path,
    results_dir: Path,
    model_name: str = "CNN",
    config_label: str = "",
) -> Dict:
    """Analyse NoFlow logit drift across series.

    Computes per-series statistics for the NoFlow logit distribution and
    creates a combined boxplot + KDE visualisation.  The statistics are
    returned as a dict (also saved as JSON) so they can be used for
    adaptive thresholding later.

    Returns:
        Dict mapping series name -> {mean, std, median, q5, q25, q75, q95,
        n_samples} for the NoFlow class, plus the same for Flow if present.
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from datetime import datetime

    model_dir = save_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Collect logits per series (merge splits)
    series_logits: Dict[str, Dict[str, list]] = {}  # series -> {noflow: [], flow: []}

    for split_name, split_tuple in splits_data.items():
        patches, labels, series_per_patch = split_tuple[0], split_tuple[1], split_tuple[2]
        dense_arr = split_tuple[3] if len(split_tuple) > 3 else None
        if len(patches) == 0:
            continue
        logits = _collect_logits(model, patches, labels, device, dense=dense_arr)
        series_arr = np.array(series_per_patch)

        for sname in np.unique(series_arr):
            if sname not in series_logits:
                series_logits[sname] = {'noflow': [], 'flow': [], 'split': split_name}
            mask = series_arr == sname
            series_logits[sname]['noflow'].extend(logits[mask & (labels == 0)].tolist())
            series_logits[sname]['flow'].extend(logits[mask & (labels == 1)].tolist())

    if not series_logits:
        return {}

    # Sort chronologically
    def _parse_date(s):
        for fmt in ('%d.%m.%Y', '%d.%m.%y'):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return datetime.max

    sorted_series = sorted(series_logits.keys(), key=_parse_date)

    # Compute statistics
    def _stats(arr):
        a = np.array(arr)
        if len(a) == 0:
            return None
        return {
            'mean': float(np.mean(a)),
            'std': float(np.std(a)),
            'median': float(np.median(a)),
            'q5': float(np.percentile(a, 5)),
            'q25': float(np.percentile(a, 25)),
            'q75': float(np.percentile(a, 75)),
            'q95': float(np.percentile(a, 95)),
            'min': float(np.min(a)),
            'max': float(np.max(a)),
            'n_samples': len(a),
        }

    drift_stats: Dict[str, Dict] = {}
    for sname in sorted_series:
        entry: Dict = {'split': series_logits[sname]['split']}
        nf_stats = _stats(series_logits[sname]['noflow'])
        fl_stats = _stats(series_logits[sname]['flow'])
        if nf_stats:
            entry['noflow'] = nf_stats
        if fl_stats:
            entry['flow'] = fl_stats
        drift_stats[sname] = entry

    # Save stats JSON
    drift_json = results_dir / 'logit_drift_stats.json'
    with open(drift_json, 'w') as f:
        json.dump(drift_stats, f, indent=2)
    print(f"  Logit drift stats saved to {drift_json}")

    # --- Visualisation ---
    # Two rows: row 0 = NoFlow, row 1 = Flow
    # Left: box+strip plot, Right: KDE overlay
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [1, 1.3]})

    split_colors_map = {'Train': '#2ca02c', 'Val': '#ff7f0e', 'Test': '#1f77b4'}

    for row, label_name in enumerate(['noflow', 'flow']):
        ax_box = axes[row, 0]
        ax_kde = axes[row, 1]

        box_data = []
        box_labels = []
        box_colors = []
        for sname in sorted_series:
            arr = series_logits[sname][label_name]
            if len(arr) == 0:
                continue
            box_data.append(arr)
            split = series_logits[sname]['split']
            box_labels.append(f"{sname}\n({split}, n={len(arr)})")
            box_colors.append(split_colors_map.get(split, 'grey'))

        if not box_data:
            ax_box.set_visible(False)
            ax_kde.set_visible(False)
            continue

        # Boxplot
        bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.6,
                            medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax_box.set_xticklabels(box_labels, fontsize=8, rotation=30, ha='right')
        ax_box.axhline(0, color='grey', linestyle='--', alpha=0.5, label='threshold=0')
        ax_box.set_ylabel('Logit value')
        class_title = 'NoFlow' if label_name == 'noflow' else 'Flow'
        ax_box.set_title(f'{class_title} — Logit Distribution per Series', fontweight='bold')
        ax_box.legend(fontsize=7)

        # KDE overlay
        for sname in sorted_series:
            arr = np.array(series_logits[sname][label_name])
            if len(arr) < 2:
                continue
            split = series_logits[sname]['split']
            # Simple KDE using histogram density
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(arr)
                x_grid = np.linspace(arr.min() - 1, arr.max() + 1, 200)
                ax_kde.plot(x_grid, kde(x_grid),
                            label=f'{sname} ({split})', linewidth=1.5)
            except np.linalg.LinAlgError:
                # Degenerate case (all same value)
                ax_kde.axvline(arr[0], label=f'{sname} ({split})', linewidth=1.5)

        ax_kde.axvline(0, color='grey', linestyle='--', alpha=0.5)
        ax_kde.set_xlabel('Logit value')
        ax_kde.set_ylabel('Density')
        ax_kde.set_title(f'{class_title} — Logit KDE per Series', fontweight='bold')
        ax_kde.legend(fontsize=7)

    # Add summary table below
    fig.suptitle(
        f'Logit Drift Analysis ({config_label})',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    fname = model_dir / f'logit_drift_{config_label}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Logit drift plot saved to {fname}")

    # Print summary table
    print(f"\n  {'Series':<15} {'Split':<6} {'NoFlow Mean':>12} {'NoFlow Std':>11} "
          f"{'NoFlow Med':>11} {'NoFlow q5':>10} {'NoFlow q95':>11} {'n':>6}")
    print(f"  {'-' * 90}")
    for sname in sorted_series:
        entry = drift_stats[sname]
        nf = entry.get('noflow')
        if nf:
            print(f"  {sname:<15} {entry['split']:<6} {nf['mean']:>12.3f} {nf['std']:>11.3f} "
                  f"{nf['median']:>11.3f} {nf['q5']:>10.3f} {nf['q95']:>11.3f} {nf['n_samples']:>6d}")

    return drift_stats


def _save_plots(
    train_losses: list,
    val_losses: list,
    metrics: Dict,
    save_dir: Path,
    model_name: str = "MLP",
    model_config: dict = None,
    series_name: Optional[str] = None,
) -> None:
    """Save training loss curves, confusion matrix, and ROC curve.

    Args:
        series_name: If given, plots are saved with series-specific filenames
            and titles. Loss curves are skipped for per-series plots.
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    if model_config is None:
        model_config = {}

    # Create model-specific subfolder
    model_dir = save_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    config_label = _build_config_label(model_name, model_config)
    file_suffix = f"_{series_name}" if series_name else ""
    title_suffix = f" [{series_name}]" if series_name else ""

    # Loss curves (only for overall, not per-series)
    if series_name is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (BCEWithLogits)')
        ax.set_title(f'Training and Validation Loss ({config_label})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(model_dir / f'loss_curves_{config_label}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    if not metrics:
        return

    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Leakage', 'Leakage'])
        ax.set_yticklabels(['No Leakage', 'Leakage'])
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

        # Build detailed title with config info
        title_lines = [f'Confusion Matrix - {model_name}{title_suffix}']
        if 'hidden_dims' in model_config:
            title_lines.append(
                f'hidden={model_config["hidden_dims"]}, '
                f'dropout={model_config.get("dropout", "?")}, '
                f'lr={model_config.get("lr", "?")}'
            )
        ax.set_title('\n'.join(title_lines), fontsize=11)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im)
        fig.savefig(model_dir / f'confusion_matrix_{config_label}{file_suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ROC curve
    if 'fpr' in metrics and 'tpr' in metrics:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(metrics['fpr'], metrics['tpr'],
                label=f"ROC (AUC = {metrics.get('roc_auc', 0):.3f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {config_label}{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(model_dir / f'roc_curve_{config_label}{file_suffix}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Plots saved to {model_dir}")
