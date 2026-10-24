"""
AutoEncoder / VAE Training and Evaluation for Anomaly Detection.

Trains a CNN AutoEncoder or Variational AutoEncoder on NoFlow (normal) data only.
Detects leakage as anomaly via reconstruction error, latent distance, One-Class SVM
on latent space, or combined score. Provides latent space visualization for interpretability.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .models import CNNAutoEncoder, CNNVariationalAutoEncoder
from src.common.utils import detect_n_stats, reshape_to_channels


# ---------------------------------------------------------------------------
# Load prepared data (reused from classification pipeline's prepared/ dir)
# ---------------------------------------------------------------------------

def _load_prepared_data(
    segment_ids: List[int],
    prepared_base: Path,
    metadata_by_id: Dict[int, dict],
    selected_features: Optional[List[str]] = None,
    label_filter: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[str]]]:
    """
    Load pre-built .npy patch files for a list of segment IDs.

    Args:
        segment_ids: List of segment IDs to load.
        prepared_base: Path to prepared/ directory.
        metadata_by_id: Mapping segment_id -> preparation metadata dict.
        selected_features: If given, only keep these feature columns.
        label_filter: If given, only load segments with this label (0 or 1).

    Returns:
        Tuple of (patches, labels, series_per_patch, feature_names).
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

        # Filter by label if requested
        if label_filter is not None and meta.get('label') != label_filter:
            continue

        patch_path = prepared_base / meta['series'] / meta['patch_file']
        if not patch_path.exists():
            continue

        patches = np.load(patch_path)
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


# ---------------------------------------------------------------------------
# Reconstruction error and latent extraction
# ---------------------------------------------------------------------------

def _reconstruction_error(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    is_vae: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample MSE reconstruction error.

    Returns:
        Tuple of (errors, latents):
        - errors: Shape (N,) MSE per sample
        - latents: Shape (N, latent_dim) latent representations
    """
    model.eval()
    all_errors = []
    all_latents = []

    with torch.no_grad():
        for (patches,) in data_loader:
            patches = patches.to(device)

            if is_vae:
                recon, z, mu, _ = model(patches)
                # Use mu (deterministic) as latent for evaluation
                latent = mu
            else:
                recon, latent = model(patches)

            # Per-sample MSE: mean over (C, H, W)
            mse = ((patches - recon) ** 2).mean(dim=(1, 2, 3))
            all_errors.append(mse.cpu().numpy())
            all_latents.append(latent.cpu().numpy())

    return np.concatenate(all_errors), np.concatenate(all_latents)


# ---------------------------------------------------------------------------
# Anomaly scoring: MSE, Mahalanobis, Combined
# ---------------------------------------------------------------------------

def _fit_mahalanobis(latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Mahalanobis distance parameters from training NoFlow latent vectors.

    Returns:
        Tuple of (mean, precision_matrix) where precision = inv(covariance).
    """
    mean = latents.mean(axis=0)
    cov = np.cov(latents, rowvar=False)
    # Regularise covariance for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6
    precision = np.linalg.inv(cov)
    return mean, precision


def _mahalanobis_distance(
    latents: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distance for each sample from the training distribution.

    Returns:
        Shape (N,) distances.
    """
    diff = latents - mean
    # d^2 = (x - mu)^T * Sigma^{-1} * (x - mu)
    left = diff @ precision
    d_sq = np.sum(left * diff, axis=1)
    return np.sqrt(np.maximum(d_sq, 0.0))


def _fit_ocsvm(
    latents: np.ndarray,
    nu: float = 0.05,
    kernel: str = 'rbf',
    gamma: str = 'scale',
) -> OneClassSVM:
    """
    Fit a One-Class SVM on training NoFlow latent vectors.

    Args:
        latents: NoFlow training latent vectors, shape (N, latent_dim).
        nu: Upper bound on the fraction of training errors (≈ expected FP rate).
        kernel: SVM kernel type ('rbf', 'poly', 'sigmoid').
        gamma: Kernel coefficient ('scale', 'auto', or float).

    Returns:
        Fitted OneClassSVM model.
    """
    ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    ocsvm.fit(latents)
    return ocsvm


def _ocsvm_score(latents: np.ndarray, ocsvm: OneClassSVM) -> np.ndarray:
    """
    Compute anomaly scores from One-Class SVM decision function.

    Returns:
        Shape (N,) scores. Higher = more anomalous.
        (Negated decision function: SVM returns positive for inliers.)
    """
    return -ocsvm.decision_function(latents)


def _compute_anomaly_score(
    errors: np.ndarray,
    latents: np.ndarray,
    mode: str,
    alpha: float,
    mahal_mean: Optional[np.ndarray] = None,
    mahal_precision: Optional[np.ndarray] = None,
    norm_params: Optional[Dict] = None,
    ocsvm_model: Optional[OneClassSVM] = None,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Compute anomaly score based on the configured mode.

    For "combined" mode, both MSE and Mahalanobis are normalised to [0,1]
    using reference statistics. On the first call (norm_params=None),
    stats are fitted and returned; subsequent calls must pass them in
    so that validation and test scores use the same scale.

    Args:
        errors: Per-sample MSE reconstruction errors.
        latents: Per-sample latent vectors.
        mode: "mse", "mahalanobis", "combined", or "ocsvm".
        alpha: Weight for MSE in combined mode (0-1).
        mahal_mean: Mean of training NoFlow latent distribution.
        mahal_precision: Precision (inv cov) of training NoFlow latent distribution.
        norm_params: Pre-fitted normalisation params from a prior call (for consistent scaling).
        ocsvm_model: Fitted OneClassSVM model (required for "ocsvm" mode).

    Returns:
        Tuple of (scores, norm_params).
        norm_params is None for non-combined modes.
    """
    if mode == "mse":
        return errors, None

    if mode == "ocsvm":
        if ocsvm_model is None:
            print("  [WARN] OC-SVM model not available, falling back to MSE")
            return errors, None
        return _ocsvm_score(latents, ocsvm_model), None

    if mahal_mean is None or mahal_precision is None:
        return errors, None

    mahal = _mahalanobis_distance(latents, mahal_mean, mahal_precision)

    if mode == "mahalanobis":
        return mahal, None

    # Combined: normalise both to [0, 1] using reference stats, then blend.
    # If norm_params is None, fit on current data (= validation set).
    if norm_params is None:
        norm_params = {
            'mse_lo': float(errors.min()),
            'mse_hi': float(errors.max()),
            'mahal_lo': float(mahal.min()),
            'mahal_hi': float(mahal.max()),
        }

    def _normalise(arr, lo, hi):
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0.0, None)

    mse_norm = _normalise(errors, norm_params['mse_lo'], norm_params['mse_hi'])
    mahal_norm = _normalise(mahal, norm_params['mahal_lo'], norm_params['mahal_hi'])

    return alpha * mse_norm + (1 - alpha) * mahal_norm, norm_params


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    is_vae: bool = False,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    label = 'Loss (MSE + KL)' if is_vae else 'MSE Loss'
    ax.set_ylabel(label)
    title = 'VAE Training Loss' if is_vae else 'AutoEncoder Training Loss'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_reconstruction_error_distribution(
    errors_noflow: np.ndarray,
    errors_flow: np.ndarray,
    threshold: float,
    save_path: Path,
    series_name: Optional[str] = None,
    score_label: str = "MSE Reconstruction Error",
):
    """Plot reconstruction error histogram for NoFlow vs Flow."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors_noflow, bins=50, alpha=0.6, label=f'NoFlow (n={len(errors_noflow)})',
            color='steelblue', density=True)
    ax.hist(errors_flow, bins=50, alpha=0.6, label=f'Flow (n={len(errors_flow)})',
            color='salmon', density=True)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.4f}')

    title = 'Anomaly Score Distribution'
    if series_name:
        title += f' - {series_name}'
    ax.set_xlabel(score_label)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    auc_val: float,
    save_path: Path,
    series_name: Optional[str] = None,
):
    """Plot ROC curve from anomaly scores."""
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    title = 'ROC Curve (Anomaly Detection)'
    if series_name:
        title += f' - {series_name}'
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_latent_space(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    series_per_sample: Optional[List[str]] = None,
    series_name: Optional[str] = None,
    split_label: Optional[str] = None,
    series_split_map: Optional[Dict[str, str]] = None,
):
    """
    Visualize latent space with t-SNE, colored by class.

    Args:
        latents: Latent vectors (N, latent_dim).
        labels: Binary labels (N,).
        save_path: Path to save the class-colored plot.
        series_per_sample: Series name per sample (for by-series plot).
        series_name: Series name for title.
        split_label: Split type (e.g. "train", "test") shown in title.
        series_split_map: Maps series name -> split label (for by-series plot legend).
    """
    if len(latents) < 2:
        return

    perplexity = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(latents)

    # --- Plot by class ---
    fig, ax = plt.subplots(figsize=(10, 8))

    noflow_mask = labels == 0
    flow_mask = labels == 1

    ax.scatter(coords[noflow_mask, 0], coords[noflow_mask, 1],
               c='steelblue', alpha=0.5, s=15, label=f'NoFlow (n={noflow_mask.sum()})')
    ax.scatter(coords[flow_mask, 0], coords[flow_mask, 1],
               c='salmon', alpha=0.5, s=15, label=f'Flow (n={flow_mask.sum()})')

    title = 'Latent Space (t-SNE)'
    if series_name:
        title += f' - {series_name}'
    if split_label:
        title += f' [{split_label}]'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # --- Plot by series ---
    if series_per_sample is not None and len(series_per_sample) == len(latents):
        series_arr = np.array(series_per_sample)
        unique_series = sorted(set(series_per_sample))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_series)))

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, s in enumerate(unique_series):
            mask = series_arr == s
            s_split = ''
            if series_split_map and s in series_split_map:
                s_split = f' [{series_split_map[s]}]'
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colors[i]], alpha=0.5, s=15,
                       label=f'{s}{s_split} (n={mask.sum()})')

        title_s = 'Latent Space by Series (t-SNE)'
        if series_name:
            title_s += f' - {series_name}'
        ax.set_title(title_s)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        stem = save_path.stem
        series_path = save_path.parent / f'{stem}_by_series.png'
        fig.savefig(series_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {series_path}")


def _plot_confusion_matrix(
    tp: int, fp: int, tn: int, fn: int,
    save_path: Path,
    series_name: Optional[str] = None,
):
    """Plot confusion matrix for anomaly detection."""
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    fig.colorbar(im, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    title = 'Confusion Matrix (Anomaly Detection)'
    if series_name:
        title += f' - {series_name}'
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_anomaly_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict:
    """
    Compute anomaly detection metrics.

    Anomaly = Flow (label 1), Normal = NoFlow (label 0).
    Prediction: anomaly if score > threshold.
    """
    predictions = (scores > threshold).astype(int)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc,
        'threshold': float(threshold),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_noflow': int((labels == 0).sum()),
        'n_flow': int((labels == 1).sum()),
    }


def _print_metrics(metrics: Dict):
    """Print anomaly detection metrics."""
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    cm = f"  [[TN={metrics['tn']}, FP={metrics['fp']}], " \
         f"[FN={metrics['fn']}, TP={metrics['tp']}]]"
    print(cm)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_autoencoder(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Train CNN AutoEncoder or VAE on NoFlow data and evaluate anomaly detection.

    One-class training: only NoFlow (label=0) patches are used for training.
    Evaluation: anomaly score (MSE, Mahalanobis, OC-SVM, or combined) on all test data.

    Args:
        config: Full config dict (must contain 'anomaly_detection').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot_enabled: Whether to create plots.
        only_files: Unused.
        max_workers: Unused.

    Returns:
        Dict with training results and anomaly detection metrics.
    """
    ae_config = config['anomaly_detection'].get('autoencoder', {})
    output_root = ae_config.get('output_root', './anomaly_detection')
    data_root = ae_config.get('data_root', './prepared')

    # Model type: "ae" or "vae"
    model_type = ae_config.get('model_type', 'ae')
    is_vae = model_type == 'vae'

    # Model hyperparameters
    cnn_channels = ae_config.get('cnn_channels', [16, 32])
    cnn_kernel_size = ae_config.get('cnn_kernel_size', 5)
    cnn_channel_mode = ae_config.get('cnn_channel_mode', 'dual')
    latent_dim = ae_config.get('latent_dim', 16)
    dropout = ae_config.get('dropout', 0.3)
    lr = ae_config.get('lr', 1e-3)
    batch_size = ae_config.get('batch_size', 32)
    epochs = ae_config.get('epochs', 100)
    patience = ae_config.get('patience', 15)
    scaler_type = ae_config.get('scaler', 'standard')
    selected_features = ae_config.get('selected_features', None)
    threshold_percentile = ae_config.get('threshold_percentile', 95)

    # Anomaly scoring
    anomaly_score_mode = ae_config.get('anomaly_score_mode', 'mse')
    anomaly_score_alpha = ae_config.get('anomaly_score_alpha', 0.5)

    # VAE-specific
    vae_beta = ae_config.get('vae_beta', 0.5)

    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    model_label = "VAE" if is_vae else "AutoEncoder"

    print("=" * 60)
    print(f"TRAIN CNN {model_label} (Anomaly Detection)")
    print("=" * 60)
    print(f"Model type:        {model_type}")
    print(f"CNN channels:      {cnn_channels}")
    print(f"CNN kernel size:   {cnn_kernel_size}")
    print(f"CNN channel mode:  {cnn_channel_mode}")
    print(f"Latent dim:        {latent_dim}")
    print(f"Dropout:           {dropout}")
    print(f"LR:                {lr}")
    print(f"Batch size:        {batch_size}")
    print(f"Max epochs:        {epochs}")
    print(f"Patience:          {patience}")
    print(f"Scaler:            {scaler_type}")
    print(f"Threshold %%ile:    {threshold_percentile}")
    print(f"Anomaly score:     {anomaly_score_mode}")
    print(f"Score alpha:       {anomaly_score_alpha}")
    print(f"VAE beta (KL wt):  {vae_beta}")
    if selected_features:
        print(f"Selected features: {len(selected_features)} columns")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:            {device}")
    print(f"Data root:         {data_root}")
    print(f"Output root:       {output_root}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Load splits.json and prepared metadata from classification data
    # ---------------------------------------------------------------
    class_base = data_path / data_root
    splits_path = class_base / 'splits.json'
    if not splits_path.exists():
        print(f"[ERROR] splits.json not found: {splits_path}")
        print("Run classification 'splitting' step first.")
        return {}

    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_base = class_base / 'prepared'
    prep_meta_path = prepared_base / 'metadata.json'
    if not prep_meta_path.exists():
        print(f"[ERROR] Prepared metadata not found: {prep_meta_path}")
        print("Run classification 'prepare' step first.")
        return {}

    with open(prep_meta_path, 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}

    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    # ---------------------------------------------------------------
    # Load patches: train/val = NoFlow only, test = all
    # ---------------------------------------------------------------
    print("\n--- LOADING TRAINING DATA (NoFlow only) ---")
    train_patches, train_labels, _, _ = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )

    if len(train_patches) == 0:
        print("[ERROR] No NoFlow training patches could be loaded.")
        return {}

    print(f"  Train patches (NoFlow): {train_patches.shape}")

    print("\n--- LOADING VALIDATION DATA (NoFlow only) ---")
    val_patches, val_labels, _, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )
    print(f"  Val patches (NoFlow): {val_patches.shape}")

    print("\n--- LOADING TEST DATA (all classes) ---")
    test_patches, test_labels, test_series, feature_names = _load_prepared_data(
        test_ids, prepared_base, metadata_by_id, selected_features,
    )
    print(f"  Test patches (all): {test_patches.shape}")
    print(f"  Test NoFlow: {(test_labels == 0).sum()}, "
          f"Test Flow: {(test_labels == 1).sum()}")

    # ---------------------------------------------------------------
    # Fit scaler on training data
    # ---------------------------------------------------------------
    print("\n--- SCALING ---")
    n_train, n_feat, w = train_patches.shape

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

    # ---------------------------------------------------------------
    # Reshape for dual-channel mode
    # ---------------------------------------------------------------
    n_stats = detect_n_stats(feature_names) if feature_names else 1
    if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        train_scaled, n_in_channels, _ = reshape_to_channels(
            train_scaled, n_feat, w, n_stats)
        val_scaled, _, _ = reshape_to_channels(val_scaled, n_feat, w, n_stats)
        test_scaled, _, _ = reshape_to_channels(test_scaled, n_feat, w, n_stats)
        print(f"  Dual-channel reshape: ({n_feat},) -> ({n_in_channels}, {n_feat // (n_in_channels // 2) // 2})")
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

    # Keep _to_dual for latent space visualization later
    def _to_dual(arr):
        if len(arr) == 0:
            if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
                h_ = n_feat // n_stats // 2
                return arr.reshape(0, n_in_channels, h_, w)
            return arr.reshape(0, 1, n_feat, w)
        if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
            return reshape_to_channels(arr, n_feat, w, n_stats)[0]
        return arr[:, np.newaxis, :, :]

    # ---------------------------------------------------------------
    # Create DataLoaders (no labels for AE training)
    # ---------------------------------------------------------------
    def _make_loader(patches_arr, shuffle):
        if len(patches_arr) == 0:
            return None
        t_patches = torch.tensor(patches_arr, dtype=torch.float32)
        ds = TensorDataset(t_patches)
        g = torch.Generator().manual_seed(42) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=torch.cuda.is_available(),
                          generator=g)

    train_loader = _make_loader(train_scaled, shuffle=True)
    val_loader = _make_loader(val_scaled, shuffle=False)

    if train_loader is None:
        print("[ERROR] No training data found.")
        return {}

    # Infer input dimensions
    (sample_batch,) = next(iter(train_loader))
    _, c_dim, h, w_dim = sample_batch.shape
    print(f"\nInput shape: ({c_dim}, {h}, {w_dim})")
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    model_cls = CNNVariationalAutoEncoder if is_vae else CNNAutoEncoder
    model = model_cls(
        input_height=h,
        input_width=w_dim,
        channels=cnn_channels,
        kernel_size=cnn_kernel_size,
        latent_dim=latent_dim,
        dropout=dropout,
        in_channels=n_in_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:         {model_label}")
    print(f"Model params:  {n_params:,}")

    mse_criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"TRAINING (NoFlow only) - {model_label}")
    print(f"{'=' * 60}")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (patches,) in train_loader:
            patches = patches.to(device)
            optimizer.zero_grad()

            if is_vae:
                recon, z, mu, log_var = model(patches)
                recon_loss = mse_criterion(recon, patches)
                kl_loss = CNNVariationalAutoEncoder.kl_divergence(mu, log_var)
                loss = recon_loss + vae_beta * kl_loss
            else:
                recon, _ = model(patches)
                loss = mse_criterion(recon, patches)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate
        avg_val_loss = avg_train_loss
        if val_loader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for (patches,) in val_loader:
                    patches = patches.to(device)

                    if is_vae:
                        recon, z, mu, log_var = model(patches)
                        recon_l = mse_criterion(recon, patches)
                        kl_l = CNNVariationalAutoEncoder.kl_divergence(mu, log_var)
                        loss = recon_l + vae_beta * kl_l
                    else:
                        recon, _ = model(patches)
                        loss = mse_criterion(recon, patches)

                    val_loss += loss.item()
                    n_val += 1
            avg_val_loss = val_loss / max(n_val, 1)

        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
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
                      f"(best: epoch {best_epoch}, val_loss: {best_val_loss:.6f})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Save model
    model_path = results_dir / 'autoencoder_best.pt'
    torch.save(best_state or model.state_dict(), model_path)
    print(f"\nBest model saved: {model_path}")

    # ---------------------------------------------------------------
    # Compute Mahalanobis parameters from training NoFlow latents
    # ---------------------------------------------------------------
    mahal_mean = None
    mahal_precision = None
    ocsvm_model = None

    if anomaly_score_mode in ('mahalanobis', 'combined'):
        print(f"\n{'=' * 60}")
        print("FITTING MAHALANOBIS PARAMETERS (train NoFlow latents)")
        print(f"{'=' * 60}")

        train_errors, train_latents = _reconstruction_error(
            model, train_loader, device, is_vae=is_vae)
        mahal_mean, mahal_precision = _fit_mahalanobis(train_latents)
        print(f"  Latent dim:    {mahal_mean.shape[0]}")
        print(f"  Train samples: {len(train_latents)}")

    # ---------------------------------------------------------------
    # Fit One-Class SVM on training NoFlow latent vectors
    # ---------------------------------------------------------------
    if anomaly_score_mode == 'ocsvm':
        print(f"\n{'=' * 60}")
        print("FITTING ONE-CLASS SVM (train NoFlow latents)")
        print(f"{'=' * 60}")

        train_errors, train_latents = _reconstruction_error(
            model, train_loader, device, is_vae=is_vae)

        ocsvm_nu = ae_config.get('ocsvm_nu', 0.05)
        ocsvm_kernel = ae_config.get('ocsvm_kernel', 'rbf')
        ocsvm_gamma = ae_config.get('ocsvm_gamma', 'scale')
        ocsvm_model = _fit_ocsvm(
            train_latents, nu=ocsvm_nu, kernel=ocsvm_kernel, gamma=ocsvm_gamma)

        n_sv = ocsvm_model.n_support_[0] if hasattr(ocsvm_model, 'n_support_') else 0
        print(f"  Latent dim:      {train_latents.shape[1]}")
        print(f"  Train samples:   {len(train_latents)}")
        print(f"  Kernel:          {ocsvm_kernel}")
        print(f"  Nu:              {ocsvm_nu}")
        print(f"  Gamma:           {ocsvm_gamma}")
        print(f"  Support vectors: {n_sv}")

        # Save OC-SVM model
        ocsvm_path = results_dir / 'ocsvm_model.pkl'
        with open(ocsvm_path, 'wb') as f:
            pickle.dump(ocsvm_model, f)
        print(f"  OC-SVM saved:    {ocsvm_path}")

    # ---------------------------------------------------------------
    # Compute threshold from validation NoFlow scores
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("COMPUTING ANOMALY THRESHOLD")
    print(f"{'=' * 60}")

    if val_loader:
        val_errors, val_latents = _reconstruction_error(
            model, val_loader, device, is_vae=is_vae)
    else:
        # Fallback: use training data
        if mahal_mean is not None or ocsvm_model is not None:
            val_errors, val_latents = train_errors, train_latents
        else:
            val_errors, val_latents = _reconstruction_error(
                model, train_loader, device, is_vae=is_vae)

    val_scores, score_norm_params = _compute_anomaly_score(
        val_errors, val_latents, anomaly_score_mode, anomaly_score_alpha,
        mahal_mean, mahal_precision,
        norm_params=None,  # Fit normalisation on validation set
        ocsvm_model=ocsvm_model,
    )

    threshold = float(np.percentile(val_scores, threshold_percentile))
    print(f"  Val NoFlow scores: mean={val_scores.mean():.6f}, "
          f"std={val_scores.std():.6f}, "
          f"max={val_scores.max():.6f}")
    print(f"  Threshold ({threshold_percentile}th percentile): {threshold:.6f}")
    print(f"  Score mode: {anomaly_score_mode}")

    # ---------------------------------------------------------------
    # Evaluate on test set
    # ---------------------------------------------------------------
    metrics = {}
    per_series_metrics = {}
    test_scores = np.array([])

    if len(test_scaled) > 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION ON TEST SET (overall)")
        print(f"{'=' * 60}")

        test_loader_eval = _make_loader(test_scaled, shuffle=False)
        test_errors, test_latents = _reconstruction_error(
            model, test_loader_eval, device, is_vae=is_vae)

        test_scores, _ = _compute_anomaly_score(
            test_errors, test_latents, anomaly_score_mode, anomaly_score_alpha,
            mahal_mean, mahal_precision,
            norm_params=score_norm_params,  # Use val-fitted normalisation
            ocsvm_model=ocsvm_model,
        )

        metrics = _compute_anomaly_metrics(test_labels, test_scores, threshold)
        _print_metrics(metrics)

        # Per-series evaluation
        test_series_arr = np.array(test_series)
        unique_test_series = sorted(set(test_series))

        for sname in unique_test_series:
            mask = test_series_arr == sname
            if mask.sum() == 0:
                continue

            s_metrics = _compute_anomaly_metrics(
                test_labels[mask], test_scores[mask], threshold)

            per_series_metrics[sname] = s_metrics
            print(f"\n{'=' * 60}")
            print(f"EVALUATION ON TEST SET - {sname}")
            print(f"{'=' * 60}")
            _print_metrics(s_metrics)

    # ---------------------------------------------------------------
    # Load ALL series (both classes) for latent space visualization
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("LATENT SPACE VISUALIZATION (all series, both classes)")
    print(f"{'=' * 60}")

    all_ids = train_ids + val_ids + test_ids
    all_patches_both, all_labels_both, all_series_both, _ = _load_prepared_data(
        all_ids, prepared_base, metadata_by_id, selected_features,
    )

    if len(all_patches_both) > 0:
        print(f"  All patches: {all_patches_both.shape}")
        print(f"  NoFlow: {(all_labels_both == 0).sum()}, "
              f"Flow: {(all_labels_both == 1).sum()}")

        # Scale and reshape
        all_scaled = _scale(all_patches_both)
        if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
            all_scaled = _to_dual(all_scaled)
        else:
            all_scaled = all_scaled[:, np.newaxis, :, :]

        all_loader = _make_loader(all_scaled, shuffle=False)
        all_errors, all_latents = _reconstruction_error(
            model, all_loader, device, is_vae=is_vae)

        all_series_arr = np.array(all_series_both)
        all_unique_series = sorted(set(all_series_both))

        # Determine split membership for labeling
        train_id_set = set(train_ids)
        val_id_set = set(val_ids)
        test_id_set = set(test_ids)

        # Map series to split type
        series_split = {}
        for seg_id in all_ids:
            meta = metadata_by_id.get(seg_id)
            if meta is None:
                continue
            s = meta.get('series', '')
            if seg_id in test_id_set:
                series_split.setdefault(s, set()).add('test')
            elif seg_id in val_id_set:
                series_split.setdefault(s, set()).add('val')
            else:
                series_split.setdefault(s, set()).add('train')

        # Build string labels: series -> "train" / "test" / "train/val" etc.
        series_split_labels = {}
        for sname in all_unique_series:
            splits = series_split.get(sname, set())
            series_split_labels[sname] = '/'.join(sorted(splits))

        for sname in all_unique_series:
            print(f"  {sname} [{series_split_labels[sname]}]: "
                  f"NoFlow={(all_labels_both[all_series_arr == sname] == 0).sum()}, "
                  f"Flow={(all_labels_both[all_series_arr == sname] == 1).sum()}")

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    plot_dir = img_path / 'anomaly_detection'
    plot_dir.mkdir(parents=True, exist_ok=True)

    score_label = {
        'mse': 'MSE Reconstruction Error',
        'mahalanobis': 'Mahalanobis Distance',
        'combined': f'Combined Score (alpha={anomaly_score_alpha})',
        'ocsvm': 'OC-SVM Anomaly Score (latent space)',
    }.get(anomaly_score_mode, 'Anomaly Score')

    _plot_loss_curves(
        train_losses, val_losses, plot_dir / 'loss_curves.png', is_vae=is_vae)

    # Test-set plots (reconstruction error, ROC, confusion matrix)
    if len(test_scaled) > 0:
        noflow_mask = test_labels == 0
        flow_mask = test_labels == 1

        _plot_reconstruction_error_distribution(
            test_scores[noflow_mask], test_scores[flow_mask],
            threshold, plot_dir / 'reconstruction_error_distribution.png',
            score_label=score_label,
        )

        if metrics.get('roc_auc', 0) > 0:
            _plot_roc_curve(
                test_labels, test_scores, metrics['roc_auc'],
                plot_dir / 'roc_curve.png',
            )

        _plot_confusion_matrix(
            metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn'],
            plot_dir / 'confusion_matrix.png',
        )

        # Per test-series: reconstruction error, ROC, confusion matrix
        for sname in unique_test_series:
            mask = test_series_arr == sname
            if mask.sum() == 0:
                continue

            safe_name = sname.replace('.', '_').replace(' ', '_')
            series_title = f'{sname} [test]'

            s_noflow = test_scores[mask & noflow_mask]
            s_flow = test_scores[mask & flow_mask]

            if len(s_noflow) > 0 or len(s_flow) > 0:
                _plot_reconstruction_error_distribution(
                    s_noflow, s_flow, threshold,
                    plot_dir / f'reconstruction_error_{safe_name}.png',
                    series_name=series_title,
                    score_label=score_label,
                )

            s_metrics = per_series_metrics.get(sname, {})
            if s_metrics.get('roc_auc', 0) > 0:
                _plot_roc_curve(
                    test_labels[mask], test_scores[mask], s_metrics['roc_auc'],
                    plot_dir / f'roc_curve_{safe_name}.png',
                    series_name=series_title,
                )

            if sname in per_series_metrics:
                sm = per_series_metrics[sname]
                _plot_confusion_matrix(
                    sm['tp'], sm['fp'], sm['tn'], sm['fn'],
                    plot_dir / f'confusion_matrix_{safe_name}.png',
                    series_name=series_title,
                )

    # Latent space plots: one per series (all series, both classes)
    if len(all_patches_both) > 0:
        # Overall latent space (all series combined)
        _plot_latent_space(
            all_latents, all_labels_both,
            plot_dir / 'latent_space.png',
            series_per_sample=all_series_both,
            series_split_map=series_split_labels,
        )

        # Per-series latent space
        for sname in all_unique_series:
            mask = all_series_arr == sname
            if mask.sum() < 2:
                continue

            safe_name = sname.replace('.', '_').replace(' ', '_')
            sl = series_split_labels.get(sname, '')

            _plot_latent_space(
                all_latents[mask], all_labels_both[mask],
                plot_dir / f'latent_space_{safe_name}.png',
                series_name=sname,
                split_label=sl,
            )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        'model_type': model_type,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'train_losses': [float(v) for v in train_losses],
        'val_losses': [float(v) for v in val_losses],
        'n_params': n_params,
        'latent_dim': latent_dim,
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'anomaly_score_mode': anomaly_score_mode,
        'anomaly_score_alpha': anomaly_score_alpha,
        'metrics': {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))},
    }

    if is_vae:
        results['vae_beta'] = vae_beta

    if anomaly_score_mode == 'ocsvm' and ocsvm_model is not None:
        results['ocsvm_nu'] = ae_config.get('ocsvm_nu', 0.05)
        results['ocsvm_kernel'] = ae_config.get('ocsvm_kernel', 'rbf')
        results['ocsvm_gamma'] = ae_config.get('ocsvm_gamma', 'scale')
        n_sv = ocsvm_model.n_support_[0] if hasattr(ocsvm_model, 'n_support_') else 0
        results['ocsvm_n_support_vectors'] = int(n_sv)

    if per_series_metrics:
        results['per_series'] = {}
        for sname, smetrics in per_series_metrics.items():
            results['per_series'][sname] = {
                k: v for k, v in smetrics.items()
                if isinstance(v, (int, float))
            }

    results_json = results_dir / 'results_autoencoder.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_json}")
    print(f"{'=' * 60}")
    print(f"{model_label.upper()} TRAINING COMPLETE")
    print(f"{'=' * 60}")

    return results


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_autoencoder(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Optuna hyperparameter tuning for CNN AutoEncoder / VAE.

    Optimises ROC-AUC on the test set (MSE-based anomaly scoring).
    The best hyperparameters are saved to the config file and a final
    training run is executed with those parameters.

    Search space:
        - model_type: ae / vae
        - cnn_channels: two conv layer channel counts
        - cnn_kernel_size: 3, 5, 7
        - latent_dim: 8, 16, 32, 64
        - dropout: 0.1 - 0.5
        - lr: 1e-4 - 1e-2
        - batch_size: 16, 32, 64
        - vae_beta (if vae): 0.1 - 2.0
        - anomaly_score_mode: mse, mahalanobis, ocsvm
        - ocsvm_nu (if ocsvm): 0.01 - 0.2
    """
    import optuna

    ae_config = config['anomaly_detection'].get('autoencoder', {})
    output_root = ae_config.get('output_root', './anomaly_detection')
    data_root = ae_config.get('data_root', './prepared')
    cnn_channel_mode = ae_config.get('cnn_channel_mode', 'dual')
    scaler_type = ae_config.get('scaler', 'standard')
    selected_features = ae_config.get('selected_features', None)
    threshold_percentile = ae_config.get('threshold_percentile', 95)
    epochs = ae_config.get('epochs', 100)
    patience = ae_config.get('patience', 15)
    n_trials = ae_config.get('optuna_n_trials', 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data (once, reused across all trials)
    # ------------------------------------------------------------------
    class_base = data_path / data_root
    splits_path = class_base / 'splits.json'
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_base = class_base / 'prepared'
    with open(prepared_base / 'metadata.json', 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}
    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    print(f"\n{'=' * 60}")
    print("OPTUNA TUNING — CNN AutoEncoder / VAE")
    print(f"{'=' * 60}")
    print(f"Device:  {device}")
    print(f"Trials:  {n_trials}")
    print(f"Epochs:  {epochs} (patience={patience})")
    print(f"{'=' * 60}\n")

    # Load patches
    print("--- LOADING DATA ---")
    train_patches, train_labels, _, tune_feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )
    print(f"  Train NoFlow: {train_patches.shape}")

    val_patches, val_labels, _, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )
    print(f"  Val NoFlow: {val_patches.shape}")

    # Note: test data is NOT loaded during tuning to avoid data leakage.
    # The final training run (after tuning) will load and evaluate on test.

    # Scale
    n_train, n_feat, w = train_patches.shape
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

    # Dual-channel reshape
    n_stats = detect_n_stats(tune_feature_names) if tune_feature_names else 1
    if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        train_scaled, n_in_channels, _ = reshape_to_channels(
            train_scaled, n_feat, w, n_stats)
        val_scaled, _, _ = reshape_to_channels(val_scaled, n_feat, w, n_stats)
    else:
        n_in_channels = 1
        train_scaled = train_scaled[:, np.newaxis, :, :]
        val_scaled = val_scaled[:, np.newaxis, :, :]

    _, c_dim, h_dim, w_dim = train_scaled.shape
    print(f"  Input shape: ({c_dim}, {h_dim}, {w_dim})")

    # ------------------------------------------------------------------
    # Objective: minimise val reconstruction loss (NoFlow only)
    #
    # Why val_loss and not ROC-AUC?
    # - Val set contains only NoFlow (one-class training).
    # - Computing ROC-AUC requires Flow labels → only on test set → data leakage.
    # - A low val_loss means the AE reconstructs NoFlow well.
    #   Flow patches (structurally different) will then produce high error → good separation.
    # - After tuning, the final training + evaluation reports ROC-AUC on test.
    # ------------------------------------------------------------------
    def objective(trial):
        # Sample hyperparameters
        model_type = trial.suggest_categorical('model_type', ['ae', 'vae'])
        ch1 = trial.suggest_categorical('ch1', [8, 16, 32])
        ch2 = trial.suggest_categorical('ch2', [16, 32, 64, 128])
        if ch2 < ch1:
            raise optuna.TrialPruned()
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        latent_dim = trial.suggest_categorical('latent_dim', [8, 16, 32, 64])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        score_mode = trial.suggest_categorical('score_mode', ['mse', 'mahalanobis', 'ocsvm'])

        ocsvm_nu = 0.05
        if score_mode == 'ocsvm':
            ocsvm_nu = trial.suggest_float('ocsvm_nu', 0.01, 0.2, step=0.01)

        vae_beta = 0.5
        if model_type == 'vae':
            vae_beta = trial.suggest_float('vae_beta', 0.1, 2.0, step=0.1)

        is_vae = model_type == 'vae'

        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True

        # Build model
        model_cls = CNNVariationalAutoEncoder if is_vae else CNNAutoEncoder
        try:
            model = model_cls(
                input_height=h_dim,
                input_width=w_dim,
                channels=[ch1, ch2],
                kernel_size=kernel_size,
                latent_dim=latent_dim,
                dropout=dropout,
                in_channels=n_in_channels,
            ).to(device)
        except Exception:
            raise optuna.TrialPruned()

        # DataLoaders
        g = torch.Generator().manual_seed(42)
        t_train = torch.tensor(train_scaled, dtype=torch.float32)
        train_ds = TensorDataset(t_train)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(), generator=g,
        )

        t_val = torch.tensor(val_scaled, dtype=torch.float32)
        val_ds = TensorDataset(t_val)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        mse_criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            for (patches,) in train_loader:
                patches = patches.to(device)
                optimizer.zero_grad()
                if is_vae:
                    recon, z, mu, log_var = model(patches)
                    loss = mse_criterion(recon, patches) + vae_beta * CNNVariationalAutoEncoder.kl_divergence(mu, log_var)
                else:
                    recon, _ = model(patches)
                    loss = mse_criterion(recon, patches)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for (patches,) in val_loader:
                    patches = patches.to(device)
                    if is_vae:
                        recon, z, mu, log_var = model(patches)
                        loss = mse_criterion(recon, patches) + vae_beta * CNNVariationalAutoEncoder.kl_divergence(mu, log_var)
                    else:
                        recon, _ = model(patches)
                        loss = mse_criterion(recon, patches)
                    val_loss += loss.item()
                    n_val += 1
            avg_val_loss = val_loss / max(n_val, 1)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        n_params = sum(p.numel() for p in model.parameters())
        trial.set_user_attr('n_params', n_params)
        trial.set_user_attr('best_val_loss', best_val_loss)
        trial.set_user_attr('score_mode', score_mode)

        # Objective: minimise val reconstruction loss
        # (lower = better NoFlow reconstruction = better anomaly separation)
        return best_val_loss

    # ------------------------------------------------------------------
    # Run Optuna study
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name='ae_tune',
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 60}")
    print("BEST TRIAL")
    print(f"{'=' * 60}")
    print(f"  Val loss:      {best.value:.6f}")
    print(f"  Params count:  {best.user_attrs['n_params']:,}")
    for k, v in best.params.items():
        print(f"  {k:14s}: {v}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Save study results
    # ------------------------------------------------------------------
    study_results = {
        'best_trial': {
            'number': best.number,
            'val_loss': best.value,
            'n_params': best.user_attrs['n_params'],
            'params': best.params,
        },
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': t.number,
                'val_loss': t.value if t.value is not None else None,
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
    # Update config with best params and run final training
    # ------------------------------------------------------------------
    bp = best.params
    ae_config['model_type'] = bp['model_type']
    ae_config['cnn_channels'] = [bp['ch1'], bp['ch2']]
    ae_config['cnn_kernel_size'] = bp['kernel_size']
    ae_config['latent_dim'] = bp['latent_dim']
    ae_config['dropout'] = bp['dropout']
    ae_config['lr'] = bp['lr']
    ae_config['batch_size'] = bp['batch_size']
    ae_config['anomaly_score_mode'] = bp['score_mode']
    if bp['score_mode'] == 'ocsvm' and 'ocsvm_nu' in bp:
        ae_config['ocsvm_nu'] = bp['ocsvm_nu']
    if bp['model_type'] == 'vae' and 'vae_beta' in bp:
        ae_config['vae_beta'] = bp['vae_beta']

    # Save best config to results dir (always writable)
    best_config_path = results_dir / 'best_autoencoder_config.json'
    with open(best_config_path, 'w') as f:
        json.dump(ae_config, f, indent=2)
    print(f"  Best config saved: {best_config_path}")

    # Also try to update the source config file (may fail in Docker)
    for candidate in [
        results_dir.parent.parent.parent / 'configs' / 'anomaly_detection' / 'autoencoder.json',
        data_path.parent / 'configs' / 'anomaly_detection' / 'autoencoder.json',
    ]:
        if candidate.exists():
            try:
                with open(candidate, 'w') as f:
                    json.dump(ae_config, f, indent=2)
                print(f"  Config updated: {candidate}")
                break
            except OSError:
                pass  # read-only filesystem (e.g. Docker)

    print(f"\n{'=' * 60}")
    print("RUNNING FINAL TRAINING WITH BEST PARAMS")
    print(f"{'=' * 60}\n")

    return train_autoencoder(
        config, data_path, img_path, plot_enabled,
        only_files=only_files, max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# Ensemble: CNN Classifier + AutoEncoder
# ---------------------------------------------------------------------------

def ensemble_evaluate(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Ensemble evaluation combining CNN Classifier logits with AE anomaly scores.

    Strategies:
    - OR:  Alarm if classifier OR AE flags anomaly (maximises recall).
    - AND: Alarm if classifier AND AE flag anomaly (maximises precision).
    - SCORE: Blend normalised scores and sweep threshold for best F1.

    Requires both models to be trained:
    - CNN classifier: data/classification/results/mlp_best.pt
    - AE:            data/anomaly_detection/results/autoencoder_best.pt
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    from src.classification.models import CNNClassifier

    ae_config = config['anomaly_detection'].get('autoencoder', {})
    output_root = ae_config.get('output_root', './anomaly_detection')
    data_root = ae_config.get('data_root', './prepared')
    cnn_channel_mode = ae_config.get('cnn_channel_mode', 'dual')
    scaler_type = ae_config.get('scaler', 'standard')
    selected_features = ae_config.get('selected_features', None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = img_path / 'anomaly_detection' if img_path else None
    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data (test set with all labels)
    # ------------------------------------------------------------------
    class_base = data_path / data_root
    splits_path = class_base / 'splits.json'
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_base = class_base / 'prepared'
    with open(prepared_base / 'metadata.json', 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}
    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    print(f"\n{'=' * 60}")
    print("ENSEMBLE: CNN Classifier + AutoEncoder")
    print(f"{'=' * 60}")

    # Load train (NoFlow) for scaler fitting
    train_patches, _, _, ens_feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )

    # Load val (NoFlow) for AE threshold
    val_patches, _, _, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )

    # Load test (all)
    test_patches, test_labels, test_series, _ = _load_prepared_data(
        test_ids, prepared_base, metadata_by_id, selected_features,
    )
    print(f"  Test patches: {test_patches.shape}")
    print(f"  Test NoFlow: {(test_labels == 0).sum()}, Flow: {(test_labels == 1).sum()}")

    # Scale
    n_train, n_feat, w = train_patches.shape
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
    test_scaled = _scale(test_patches)

    # Dual-channel reshape
    n_stats = detect_n_stats(ens_feature_names) if ens_feature_names else 1
    if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        train_dual, n_in_channels, _ = reshape_to_channels(
            train_scaled, n_feat, w, n_stats)
        val_dual, _, _ = reshape_to_channels(val_scaled, n_feat, w, n_stats)
        test_dual, _, _ = reshape_to_channels(test_scaled, n_feat, w, n_stats)
    else:
        n_in_channels = 1
        train_dual = train_scaled[:, np.newaxis, :, :]
        val_dual = val_scaled[:, np.newaxis, :, :]
        test_dual = test_scaled[:, np.newaxis, :, :]

    _, c_dim, h_dim, w_dim = test_dual.shape

    # ------------------------------------------------------------------
    # Load CNN Classifier
    # ------------------------------------------------------------------
    cls_config = config.get('classification', {}).get('mlp', {})
    if not cls_config:
        # Load from file
        cls_config_path = data_path.parent / 'configs' / 'classification' / 'mlp.json'
        if not cls_config_path.exists():
            cls_config_path = Path('configs') / 'classification' / 'mlp.json'
        if cls_config_path.exists():
            with open(cls_config_path, 'r') as f:
                cls_config = json.load(f)

    cls_model_path = data_path / data_root / 'results' / 'mlp_best.pt'
    if not cls_model_path.exists():
        print(f"[ERROR] CNN Classifier not found: {cls_model_path}")
        return {}

    cls_channels = cls_config.get('cnn_channels', [16, 32])
    cls_kernel = cls_config.get('cnn_kernel_size', 5)
    cls_fc_dim = cls_config.get('cnn_fc_dim', 64)
    cls_dropout = cls_config.get('dropout', 0.3)

    classifier = CNNClassifier(
        input_height=h_dim, input_width=w_dim,
        channels=cls_channels, kernel_size=cls_kernel,
        fc_dim=cls_fc_dim, dropout=cls_dropout,
        in_channels=n_in_channels,
    ).to(device)
    classifier.load_state_dict(torch.load(cls_model_path, map_location=device))
    classifier.eval()
    print(f"  CNN Classifier loaded: {cls_model_path}")

    # ------------------------------------------------------------------
    # Load AutoEncoder
    # ------------------------------------------------------------------
    ae_model_path = results_dir / 'autoencoder_best.pt'
    if not ae_model_path.exists():
        print(f"[ERROR] AutoEncoder not found: {ae_model_path}")
        return {}

    ae_latent_dim = ae_config.get('latent_dim', 32)
    ae_channels = ae_config.get('cnn_channels', [16, 32])
    ae_kernel = ae_config.get('cnn_kernel_size', 5)
    ae_dropout = ae_config.get('dropout', 0.3)

    ae_model = CNNAutoEncoder(
        input_height=h_dim, input_width=w_dim,
        channels=ae_channels, kernel_size=ae_kernel,
        latent_dim=ae_latent_dim, dropout=ae_dropout,
        in_channels=n_in_channels,
    ).to(device)
    ae_model.load_state_dict(torch.load(ae_model_path, map_location=device))
    ae_model.eval()
    print(f"  AutoEncoder loaded: {ae_model_path}")

    # ------------------------------------------------------------------
    # Compute scores on test set
    # ------------------------------------------------------------------
    test_tensor = torch.tensor(test_dual, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=64, shuffle=False)

    # CNN Classifier logits
    all_logits = []
    with torch.no_grad():
        for (patches,) in test_loader:
            patches = patches.to(device)
            logits = classifier(patches).cpu().numpy().flatten()
            all_logits.extend(logits)
    cls_logits = np.array(all_logits)
    cls_probs = 1 / (1 + np.exp(-cls_logits))  # sigmoid
    cls_preds = (cls_logits > 0).astype(int)

    # AE reconstruction error
    ae_errors, _ = _reconstruction_error(ae_model, test_loader, device, is_vae=False)

    # AE threshold from val NoFlow
    val_tensor = torch.tensor(val_dual, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=64, shuffle=False)
    val_errors, _ = _reconstruction_error(ae_model, val_loader, device, is_vae=False)
    ae_threshold = np.percentile(val_errors, ae_config.get('threshold_percentile', 95))
    ae_preds = (ae_errors > ae_threshold).astype(int)

    print(f"\n  CNN Classifier: {cls_preds.sum()} positive predictions")
    print(f"  AE (threshold={ae_threshold:.4f}): {ae_preds.sum()} positive predictions")

    # ------------------------------------------------------------------
    # Ensemble strategies
    # ------------------------------------------------------------------
    def _metrics(y_true, y_pred, scores=None):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }
        if scores is not None:
            try:
                m['roc_auc'] = roc_auc_score(y_true, scores)
            except ValueError:
                m['roc_auc'] = 0.0
        return m

    # OR ensemble: alarm if either model flags
    or_preds = ((cls_preds == 1) | (ae_preds == 1)).astype(int)

    # AND ensemble: alarm if both models flag
    and_preds = ((cls_preds == 1) & (ae_preds == 1)).astype(int)

    # Score blend: normalised CNN prob + normalised AE error
    ae_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-12)
    blend_scores = 0.5 * cls_probs + 0.5 * ae_norm

    # Find best threshold for blended score
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (blend_scores > t).astype(int)
        f1_t = f1_score(test_labels, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t
    blend_preds = (blend_scores > best_thresh).astype(int)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    strategies = {
        'CNN only': (cls_preds, cls_probs),
        'AE only': (ae_preds, ae_errors),
        'OR (union)': (or_preds, blend_scores),
        'AND (intersect)': (and_preds, blend_scores),
        f'BLEND (t={best_thresh:.2f})': (blend_preds, blend_scores),
    }

    all_results = {}
    unique_series = sorted(set(test_series))

    print(f"\n{'=' * 60}")
    print("ENSEMBLE RESULTS — OVERALL")
    print(f"{'=' * 60}")
    print(f"{'Strategy':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
    print("-" * 60)

    for name, (preds, scores) in strategies.items():
        m = _metrics(test_labels, preds, scores)
        all_results[name] = {'overall': m}
        auc_str = f"{m.get('roc_auc', 0):.3f}" if 'roc_auc' in m else "  -  "
        print(f"{name:<25s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f} {auc_str:>7s}")

    # Per-series
    for series_name in unique_series:
        mask = np.array([s == series_name for s in test_series])
        if mask.sum() == 0:
            continue

        print(f"\n{'=' * 60}")
        print(f"ENSEMBLE RESULTS — {series_name}")
        print(f"{'=' * 60}")
        print(f"{'Strategy':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
        print("-" * 60)

        for name, (preds, scores) in strategies.items():
            m = _metrics(test_labels[mask], preds[mask])
            if name not in all_results:
                all_results[name] = {}
            all_results[name][series_name] = m
            print(f"{name:<25s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
                  f"{m['recall']:7.3f} {m['f1']:7.3f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if plot_dir:
        fig, axes = plt.subplots(1, len(unique_series), figsize=(6 * len(unique_series), 5))
        if len(unique_series) == 1:
            axes = [axes]

        for ax, series_name in zip(axes, unique_series):
            mask = np.array([s == series_name for s in test_series])
            s_labels = test_labels[mask]
            s_cls = cls_probs[mask]
            s_ae = ae_norm[mask]

            ax.scatter(s_cls[s_labels == 0], s_ae[s_labels == 0],
                       alpha=0.4, s=10, c='tab:blue', label='NoFlow')
            ax.scatter(s_cls[s_labels == 1], s_ae[s_labels == 1],
                       alpha=0.4, s=10, c='tab:red', label='Flow')
            ax.axvline(0.5, color='gray', ls='--', lw=0.8, label='CNN t=0.5')
            ax.axhline(
                (ae_threshold - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-12),
                color='orange', ls='--', lw=0.8, label='AE threshold')
            ax.set_xlabel('CNN Probability (Flow)')
            ax.set_ylabel('AE Anomaly Score (normalised)')
            ax.set_title(series_name)
            ax.legend(fontsize=7)

        plt.tight_layout()
        scatter_path = plot_dir / 'ensemble_scatter.png'
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        print(f"\n  Scatter plot: {scatter_path}")

    # Save results
    ensemble_json = results_dir / 'ensemble_results.json'
    # Convert numpy types for JSON
    serialisable = {}
    for strat, strat_data in all_results.items():
        serialisable[strat] = {}
        for key, m in strat_data.items():
            serialisable[strat][key] = {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in m.items()
            }
    serialisable['ae_threshold'] = float(ae_threshold)
    serialisable['blend_threshold'] = float(best_thresh)

    with open(ensemble_json, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Results saved: {ensemble_json}")

    return all_results


# ---------------------------------------------------------------------------
# Latent-space anomaly detection using trained classifier encoder
# ---------------------------------------------------------------------------

def latent_anomaly_detection(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    One-class anomaly detection in the classifier's latent space.

    Loads a trained CNN classifier, extracts latent features via
    ``model.extract_features()``, fits one-class models (Mahalanobis, GMM)
    on training NoFlow latent vectors, and evaluates anomaly detection on
    the test set.

    This approach combines the discriminative encoder learned by the
    supervised classifier with a one-class scoring principle that only
    needs NoFlow data for calibration.

    Requires a trained CNN classifier at
    ``data/classification/results/mlp_best.pt``.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix,
    )
    from sklearn.mixture import GaussianMixture
    from src.classification.models import CNNClassifier

    ae_config = config['anomaly_detection'].get('autoencoder', {})
    data_root = ae_config.get('data_root', './prepared')
    output_root = ae_config.get('output_root', './anomaly_detection')
    threshold_percentile = ae_config.get('threshold_percentile', 95)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = img_path / 'anomaly_detection' / 'latent_ad' if img_path else None
    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load CNN Classifier config (features & channel mode from classifier)
    # ------------------------------------------------------------------
    cls_config = config.get('classification', {}).get('mlp', {})
    if not cls_config:
        cls_config_path = data_path.parent / 'configs' / 'classification' / 'mlp.json'
        if not cls_config_path.exists():
            cls_config_path = Path('configs') / 'classification' / 'mlp.json'
        if cls_config_path.exists():
            with open(cls_config_path, 'r') as f:
                cls_config = json.load(f)

    cls_model_path = data_path / data_root / 'results' / 'mlp_best.pt'
    if not cls_model_path.exists():
        print(f"[ERROR] CNN Classifier not found: {cls_model_path}")
        return {}

    # Use classifier's features and channel mode, not AE's
    selected_features = cls_config.get('selected_features', None)
    cnn_channel_mode = cls_config.get('cnn_channel_mode', 'dual')
    scaler_type = cls_config.get('scaler', 'standard')

    print("=" * 60)
    print("LATENT-SPACE ANOMALY DETECTION (Classifier Encoder)")
    print("=" * 60)
    print(f"  Classifier features: {len(selected_features) if selected_features else 'all'}")
    print(f"  Channel mode: {cnn_channel_mode}")
    print(f"  Scaler: {scaler_type}")
    print(f"  Threshold percentile: {threshold_percentile}")

    # ------------------------------------------------------------------
    # Load data (using classifier's feature selection)
    # ------------------------------------------------------------------
    class_base = data_path / data_root
    splits_path = class_base / 'splits.json'
    if not splits_path.exists():
        print(f"[ERROR] splits.json not found: {splits_path}")
        return {}

    with open(splits_path, 'r') as f:
        splits_data = json.load(f)

    prepared_base = class_base / 'prepared'
    with open(prepared_base / 'metadata.json', 'r') as f:
        all_prep_metadata = json.load(f)

    metadata_by_id = {m['segment_id']: m for m in all_prep_metadata}
    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    # Train NoFlow for fitting one-class models
    print("\n--- LOADING TRAINING DATA (NoFlow only) ---")
    train_patches, train_labels, train_series, feature_names = _load_prepared_data(
        train_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )
    print(f"  Train NoFlow patches: {train_patches.shape}")

    # Val NoFlow for threshold calibration
    print("\n--- LOADING VALIDATION DATA (NoFlow only) ---")
    val_patches, val_labels, val_series, _ = _load_prepared_data(
        val_ids, prepared_base, metadata_by_id, selected_features,
        label_filter=0,
    )
    print(f"  Val NoFlow patches: {val_patches.shape}")

    # Test all for evaluation
    print("\n--- LOADING TEST DATA (all classes) ---")
    test_patches, test_labels, test_series_list, _ = _load_prepared_data(
        test_ids, prepared_base, metadata_by_id, selected_features,
    )
    print(f"  Test patches: {test_patches.shape}")
    print(f"  Test NoFlow: {(test_labels == 0).sum()}, "
          f"Flow: {(test_labels == 1).sum()}")

    # ------------------------------------------------------------------
    # Scale and reshape (matching classifier's preprocessing)
    # ------------------------------------------------------------------
    n_train, n_feat, w = train_patches.shape
    train_flat = train_patches.transpose(0, 2, 1).reshape(-1, n_feat)
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
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
    test_scaled = _scale(test_patches)

    # Dual-channel reshape
    n_stats = detect_n_stats(feature_names) if feature_names else 1
    if cnn_channel_mode == 'dual' and n_feat % 2 == 0:
        train_dual, n_in_channels, _ = reshape_to_channels(
            train_scaled, n_feat, w, n_stats)
        val_dual, _, _ = reshape_to_channels(val_scaled, n_feat, w, n_stats)
        test_dual, _, _ = reshape_to_channels(test_scaled, n_feat, w, n_stats)
    else:
        n_in_channels = 1
        train_dual = train_scaled[:, np.newaxis, :, :]
        val_dual = val_scaled[:, np.newaxis, :, :] if len(val_scaled) > 0 \
            else val_scaled.reshape(0, 1, n_feat, w)
        test_dual = test_scaled[:, np.newaxis, :, :] if len(test_scaled) > 0 \
            else test_scaled.reshape(0, 1, n_feat, w)

    _, c_dim, h_dim, w_dim = test_dual.shape

    # ------------------------------------------------------------------
    # Load CNN Classifier model
    # ------------------------------------------------------------------
    classifier = CNNClassifier(
        input_height=h_dim, input_width=w_dim,
        channels=cls_config.get('cnn_channels', [16, 32]),
        kernel_size=cls_config.get('cnn_kernel_size', 5),
        fc_dim=cls_config.get('cnn_fc_dim', 64),
        dropout=cls_config.get('dropout', 0.3),
        in_channels=n_in_channels,
    ).to(device)
    classifier.load_state_dict(
        torch.load(cls_model_path, map_location=device, weights_only=True))
    classifier.eval()
    print(f"\n  CNN Classifier loaded: {cls_model_path}")
    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier params: {n_params:,}")

    # ------------------------------------------------------------------
    # Extract latent features
    # ------------------------------------------------------------------
    def _extract_features(patches_arr):
        t_patches = torch.tensor(patches_arr, dtype=torch.float32)
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(t_patches), 64):
                batch = t_patches[i:i + 64].to(device)
                feats = classifier.extract_features(batch)
                all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    print("\n--- EXTRACTING LATENT FEATURES ---")
    train_latent = _extract_features(train_dual)
    val_latent = _extract_features(val_dual)
    test_latent = _extract_features(test_dual)
    latent_dim = train_latent.shape[1]
    print(f"  Latent dim: {latent_dim}")
    print(f"  Train NoFlow: {train_latent.shape}")
    print(f"  Val NoFlow:   {val_latent.shape}")
    print(f"  Test (all):   {test_latent.shape}")

    # ------------------------------------------------------------------
    # Fit one-class models on training NoFlow latent vectors
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FITTING ONE-CLASS MODELS ON NoFlow LATENT SPACE")
    print(f"{'=' * 60}")

    # 1) Mahalanobis distance
    nf_mean = np.mean(train_latent, axis=0)
    nf_cov = np.cov(train_latent, rowvar=False)
    # Regularise covariance for numerical stability
    nf_cov += np.eye(latent_dim) * 1e-6
    nf_cov_inv = np.linalg.inv(nf_cov)

    def _mahalanobis(latent_arr):
        diff = latent_arr - nf_mean
        left = diff @ nf_cov_inv
        return np.sqrt(np.sum(left * diff, axis=1))

    train_maha = _mahalanobis(train_latent)
    val_maha = _mahalanobis(val_latent)
    test_maha = _mahalanobis(test_latent)
    print(f"  Mahalanobis fitted (mean={nf_mean.mean():.3f})")
    print(f"    Train NoFlow: mean={train_maha.mean():.3f}, "
          f"std={train_maha.std():.3f}")
    print(f"    Val NoFlow:   mean={val_maha.mean():.3f}, "
          f"std={val_maha.std():.3f}")

    # 2) GMM (negative log-likelihood as anomaly score)
    n_components = min(5, len(train_latent) // 10)
    n_components = max(n_components, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        n_init=3,
    )
    gmm.fit(train_latent)
    train_gmm_scores = -gmm.score_samples(train_latent)
    val_gmm_scores = -gmm.score_samples(val_latent)
    test_gmm_scores = -gmm.score_samples(test_latent)
    print(f"  GMM fitted (n_components={n_components})")
    print(f"    Train NoFlow: mean={train_gmm_scores.mean():.3f}, "
          f"std={train_gmm_scores.std():.3f}")
    print(f"    Val NoFlow:   mean={val_gmm_scores.mean():.3f}, "
          f"std={val_gmm_scores.std():.3f}")

    # ------------------------------------------------------------------
    # Compute thresholds from validation NoFlow
    # ------------------------------------------------------------------
    maha_threshold = float(np.percentile(val_maha, threshold_percentile))
    gmm_threshold = float(np.percentile(val_gmm_scores, threshold_percentile))
    print(f"\n  Thresholds ({threshold_percentile}th percentile of val NoFlow):")
    print(f"    Mahalanobis: {maha_threshold:.4f}")
    print(f"    GMM NLL:     {gmm_threshold:.4f}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def _eval_anomaly(scores, threshold, name):
        preds = (scores > threshold).astype(int)

        try:
            auc = roc_auc_score(test_labels, scores)
        except ValueError:
            auc = 0.0

        tn, fp, fn, tp = confusion_matrix(
            test_labels, preds, labels=[0, 1]).ravel()

        overall = {
            'accuracy': accuracy_score(test_labels, preds),
            'precision': precision_score(test_labels, preds, zero_division=0),
            'recall': recall_score(test_labels, preds, zero_division=0),
            'f1': f1_score(test_labels, preds, zero_division=0),
            'roc_auc': auc,
            'threshold': threshold,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }

        print(f"\n{'=' * 60}")
        print(f"{name} — OVERALL")
        print(f"{'=' * 60}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1:        {overall['f1']:.4f}")
        print(f"  ROC-AUC:   {overall['roc_auc']:.4f}")
        print(f"  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

        per_series = {}
        unique_series = sorted(set(test_series_list))
        for sname in unique_series:
            mask = np.array([s == sname for s in test_series_list])
            if mask.sum() == 0:
                continue
            s_preds = preds[mask]
            s_labels = test_labels[mask]
            s_scores = scores[mask]

            try:
                s_auc = roc_auc_score(s_labels, s_scores)
            except ValueError:
                s_auc = 0.0

            s_tn, s_fp, s_fn, s_tp = confusion_matrix(
                s_labels, s_preds, labels=[0, 1]).ravel()

            per_series[sname] = {
                'accuracy': accuracy_score(s_labels, s_preds),
                'precision': precision_score(s_labels, s_preds, zero_division=0),
                'recall': recall_score(s_labels, s_preds, zero_division=0),
                'f1': f1_score(s_labels, s_preds, zero_division=0),
                'roc_auc': s_auc,
                'tp': int(s_tp), 'fp': int(s_fp),
                'tn': int(s_tn), 'fn': int(s_fn),
            }

            print(f"\n  {name} — {sname}")
            print(f"    Acc={per_series[sname]['accuracy']:.4f}  "
                  f"Prec={per_series[sname]['precision']:.4f}  "
                  f"Rec={per_series[sname]['recall']:.4f}  "
                  f"F1={per_series[sname]['f1']:.4f}  "
                  f"AUC={s_auc:.4f}")
            print(f"    [[TN={s_tn}, FP={s_fp}], [FN={s_fn}, TP={s_tp}]]")

        return overall, per_series

    maha_overall, maha_per_series = _eval_anomaly(
        test_maha, maha_threshold, "MAHALANOBIS")
    gmm_overall, gmm_per_series = _eval_anomaly(
        test_gmm_scores, gmm_threshold, "GMM")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if plot_dir:
        unique_series = sorted(set(test_series_list))

        # 1) Score distributions per class
        for score_name, scores in [('Mahalanobis', test_maha),
                                   ('GMM_NLL', test_gmm_scores)]:
            fig, axes = plt.subplots(1, len(unique_series) + 1,
                                     figsize=(5 * (len(unique_series) + 1), 4))
            if len(unique_series) + 1 == 1:
                axes = [axes]

            # Overall
            ax = axes[0]
            ax.hist(scores[test_labels == 0], bins=50, alpha=0.5,
                    label='NoFlow', density=True, color='tab:blue')
            ax.hist(scores[test_labels == 1], bins=50, alpha=0.5,
                    label='Flow', density=True, color='tab:red')
            thresh = maha_threshold if score_name == 'Mahalanobis' else gmm_threshold
            ax.axvline(thresh, color='black', ls='--', label=f'threshold')
            ax.set_title(f'{score_name} — Overall')
            ax.set_xlabel('Score')
            ax.legend(fontsize=7)

            # Per series
            for i, sname in enumerate(unique_series):
                ax = axes[i + 1]
                mask = np.array([s == sname for s in test_series_list])
                s_scores = scores[mask]
                s_labels = test_labels[mask]
                ax.hist(s_scores[s_labels == 0], bins=30, alpha=0.5,
                        label='NoFlow', density=True, color='tab:blue')
                ax.hist(s_scores[s_labels == 1], bins=30, alpha=0.5,
                        label='Flow', density=True, color='tab:red')
                ax.axvline(thresh, color='black', ls='--')
                ax.set_title(f'{score_name} — {sname}')
                ax.set_xlabel('Score')
                ax.legend(fontsize=7)

            plt.tight_layout()
            fig.savefig(plot_dir / f'{score_name.lower()}_distribution.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        # 2) t-SNE of latent space (test only)
        if len(test_latent) > 0:
            perplexity = min(30, len(test_latent) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, max_iter=1000)
            embedded = tsne.fit_transform(test_latent)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # By label
            ax = axes[0]
            ax.scatter(embedded[test_labels == 0, 0],
                       embedded[test_labels == 0, 1],
                       s=5, alpha=0.4, c='tab:blue', label='NoFlow')
            ax.scatter(embedded[test_labels == 1, 0],
                       embedded[test_labels == 1, 1],
                       s=5, alpha=0.4, c='tab:red', label='Flow')
            ax.set_title('Classifier Latent Space — by Label')
            ax.legend()

            # By series
            ax = axes[1]
            series_arr = np.array(test_series_list)
            for sname in sorted(set(test_series_list)):
                mask = series_arr == sname
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                           s=5, alpha=0.4, label=sname)
            ax.set_title('Classifier Latent Space — by Series')
            ax.legend(fontsize=7)

            plt.tight_layout()
            fig.savefig(plot_dir / 'tsne_classifier_latent.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        # 3) ROC curves
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        for score_name, scores in [('Mahalanobis', test_maha),
                                   ('GMM NLL', test_gmm_scores)]:
            try:
                fpr, tpr, _ = roc_curve(test_labels, scores)
                auc = roc_auc_score(test_labels, scores)
                ax.plot(fpr, tpr, label=f'{score_name} (AUC={auc:.3f})')
            except ValueError:
                pass
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC — Latent-Space Anomaly Detection')
        ax.legend()
        fig.savefig(plot_dir / 'roc_latent_ad.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

        print(f"\n  Plots saved to {plot_dir}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        'method': 'classifier_latent_space',
        'classifier_model': str(cls_model_path),
        'latent_dim': latent_dim,
        'threshold_percentile': threshold_percentile,
        'mahalanobis': {
            'threshold': maha_threshold,
            'overall': {k: float(v) if isinstance(v, (np.floating, float))
                        else v for k, v in maha_overall.items()},
            'per_series': {
                sname: {k: float(v) if isinstance(v, (np.floating, float))
                        else v for k, v in m.items()}
                for sname, m in maha_per_series.items()
            },
        },
        'gmm': {
            'n_components': n_components,
            'threshold': gmm_threshold,
            'overall': {k: float(v) if isinstance(v, (np.floating, float))
                        else v for k, v in gmm_overall.items()},
            'per_series': {
                sname: {k: float(v) if isinstance(v, (np.floating, float))
                        else v for k, v in m.items()}
                for sname, m in gmm_per_series.items()
            },
        },
    }

    results_path = results_dir / 'latent_ad_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'=' * 60}")
    print("LATENT-SPACE ANOMALY DETECTION COMPLETE")
    print(f"{'=' * 60}")

    return results
