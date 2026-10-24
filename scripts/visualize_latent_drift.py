#!/usr/bin/env python
"""
Visualize NoFlow cluster drift across series in contrastive encoder latent space.

Loads the best Optuna checkpoint, extracts latents for ALL 8 series,
and produces:
  1. PCA projection: NoFlow + Leakage, colored by series (chronological)
  2. NoFlow centroid trajectory over time with covariance ellipses
  3. Per-dimension centroid drift (raw latent space, no PCA)
  4. Mahalanobis distance of each series' NoFlow centroid from the
     earliest series (= simulated temporal drift magnitude)

Run on the GPU server:
    python scripts/visualize_latent_drift.py [--checkpoint PATH]

Outputs saved to images/contrastive/drift/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contrastive.models import (
    ContrastiveCNNAutoEncoder,
    build_contrastive_model,
)
from src.contrastive.train import (
    _load_splits_and_metadata,
    _ids_for_series,
    _prepare_patches,
    _scale_patches,
    _to_channels,
    _extract_latents,
)


# Chronological series order
SERIES_ORDER = [
    '10.01.2026', '24.01.2026', '7.02.2026',
    '21.02.2026', '7.03.2026', '15.03.2026',
    '5.04.2026', '11.04.2026', '18.04.2026',
    'ha-25.04.2026', 'hb-26.04.2026',
]
SERIES_LABELS = [
    'Jan 10\n(Winter)', 'Jan 24\n(Winter)', 'Feb 7\n(Winter)',
    'Feb 21\n(Late Win.)', 'Mar 7\n(Spring)', 'Mar 15\n(Spring\nlow flow)',
    'Apr 5\n(Spring)', 'Apr 11\n(Spring)', 'Apr 18\n(Spring)',
    'Apr 25\n(Haus A)', 'Apr 26\n(Haus B)',
]
SERIES_COLORS = [
    '#1f77b4', '#2ca02c', '#17becf', '#9467bd',
    '#ff7f0e', '#d62728', '#e377c2', '#8c564b',
    '#bcbd22', '#7f7f7f', '#000000',
]


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model + metadata from a trial checkpoint.

    Works for both 'autoencoder' and 'classifier' encoder variants — the
    encoder_type is read from the checkpoint's ``hp`` dict (falls back to
    'autoencoder' for older checkpoints that predate the classifier option).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    hp = ckpt.get('hp', {})

    model = build_contrastive_model(
        encoder_type=hp.get('encoder_type', 'autoencoder'),
        input_height=ckpt.get('input_height', 9),
        input_width=ckpt.get('input_width', 32),
        in_channels=ckpt.get('n_in_channels', 2),
        channels=hp.get('cnn_channels', [32, 64, 128]),
        kernel_size=hp.get('cnn_kernel_size', 3),
        latent_dim=hp.get('latent_dim', 128),
        projection_dim=hp.get('projection_dim', 64),
        dropout=hp.get('dropout', 0.3),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, hp


def load_all_series_data(data_path: Path, cfg: dict):
    """Load and scale patches for all 8 series, returning per-series arrays."""
    data_root = cfg.get('data_root', './prepared')
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(
        data_path, data_root
    )

    selected_features = cfg.get('selected_features', None)

    # Load ALL patches across all splits
    all_ids = []
    for split in ('train', 'val', 'test'):
        all_ids.extend(splits_data['splits'].get(split, []))

    all_patches, all_labels, all_series, feat_names = _prepare_patches(
        all_ids, prepared_base, metadata_by_id, selected_features
    )

    # Scale with NoFlow-only StandardScaler (same as training)
    train_ids = splits_data['splits']['train']
    train_patches, train_labels, train_series, _ = _prepare_patches(
        train_ids, prepared_base, metadata_by_id, selected_features
    )
    (scaled_train, scaled_all), scaler = _scale_patches(
        train_patches, train_labels, [all_patches],
        scaler_type=cfg.get('scaler', 'standard'),
    )

    # Convert to dual channels
    n_feat = scaled_all.shape[1]
    w = scaled_all.shape[2]
    n_stats = 1
    dual = cfg.get('cnn_channel_mode', 'dual') == 'dual'
    chan_all, n_ch = _to_channels(scaled_all, n_feat, w, n_stats, dual)

    # Split into per-series dicts
    per_series = {}
    for s in SERIES_ORDER:
        mask = np.array(all_series) == s
        if mask.sum() == 0:
            continue
        per_series[s] = {
            'patches': chan_all[mask],
            'labels': np.array(all_labels)[mask],
        }

    return per_series


def confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    """Draw a covariance ellipse at `n_std` standard deviations."""
    from matplotlib.patches import Ellipse
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0))
    ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellip)
    return ellip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint (default: best trial from optuna_best_params.json)'
    )
    parser.add_argument('--data-path', type=str, default='data')
    parser.add_argument('--img-path', type=str, default='images')
    parser.add_argument('--config-dir', type=str, default='configs/contrastive')
    args = parser.parse_args()

    import os
    if os.getenv('DOCKER_ENV'):
        data_path = Path(os.getenv('CONTAINER_DATA_PATH', '/app/data'))
        img_path = Path(os.getenv('CONTAINER_IMG_PATH', '/app/images'))
        config_dir = Path(os.getenv('CONFIG_DIR_PATH', '/app/configs')) / 'contrastive'
    else:
        root = Path(__file__).resolve().parent.parent
        data_path = root / args.data_path
        img_path = root / args.img_path
        config_dir = root / args.config_dir

    out_dir = img_path / 'contrastive' / 'drift'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_dir / 'encoder.json') as f:
        cfg = json.load(f)

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        best_path = data_path / 'contrastive' / 'results' / 'optuna_best_params.json'
        if best_path.exists():
            best = json.load(open(best_path))
            trial_num = best['best_trial']
        else:
            trial_num = 189
        ckpt_path = data_path / 'contrastive' / 'results' / 'checkpoints' / f'trial_{trial_num:04d}.pt'

    print(f"Checkpoint: {ckpt_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, params = load_model_from_checkpoint(ckpt_path, device)
    print(f"Model loaded: latent_dim={params.get('latent_dim')}")

    # Load data
    print("Loading all series...")
    per_series = load_all_series_data(data_path, cfg)
    print(f"Series loaded: {list(per_series.keys())}")

    # Extract latents per series
    latents = {}
    for s in SERIES_ORDER:
        if s not in per_series:
            continue
        z = _extract_latents(model, per_series[s]['patches'], device)
        latents[s] = {
            'z': z,
            'labels': per_series[s]['labels'],
        }
        n_nf = (per_series[s]['labels'] == 0).sum()
        n_fl = (per_series[s]['labels'] == 1).sum()
        print(f"  {s}: {len(z)} patches ({n_nf} NoFlow, {n_fl} Flow), latent shape {z.shape}")

    # ---- Compute per-series NoFlow statistics ----
    noflow_centroids = {}
    noflow_covs = {}
    noflow_latents_all = {}
    for s in SERIES_ORDER:
        if s not in latents:
            continue
        nf_mask = latents[s]['labels'] == 0
        nf_z = latents[s]['z'][nf_mask]
        noflow_centroids[s] = nf_z.mean(axis=0)
        noflow_covs[s] = np.cov(nf_z, rowvar=False)
        noflow_latents_all[s] = nf_z

    # ---- PCA on ALL latents combined (for consistent axes) ----
    all_z = np.concatenate([latents[s]['z'] for s in SERIES_ORDER if s in latents])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_z)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # =================================================================
    # PLOT 1: PCA scatter — all patches, colored by series, shape by label
    # =================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    for i, s in enumerate(SERIES_ORDER):
        if s not in latents:
            continue
        z2d = pca.transform(latents[s]['z'])
        nf = latents[s]['labels'] == 0
        fl = latents[s]['labels'] == 1
        ax1.scatter(z2d[nf, 0], z2d[nf, 1], s=8, alpha=0.3,
                    color=SERIES_COLORS[i], marker='o', label=f'{s} NoFlow')
        if fl.sum() > 0:
            ax1.scatter(z2d[fl, 0], z2d[fl, 1], s=25, alpha=0.6,
                        color=SERIES_COLORS[i], marker='x', label=f'{s} Flow')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax1.set_title('Latent Space (PCA) — All Series\n(o = NoFlow, x = Leakage)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2, loc='best')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(out_dir / 'latent_pca_all_series.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir / 'latent_pca_all_series.png'}")

    # =================================================================
    # PLOT 2: NoFlow centroid trajectory + covariance ellipses
    # =================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    centroids_2d = []
    for i, s in enumerate(SERIES_ORDER):
        if s not in noflow_latents_all:
            continue
        nf_2d = pca.transform(noflow_latents_all[s])
        c2d = nf_2d.mean(axis=0)
        cov2d = np.cov(nf_2d, rowvar=False)
        centroids_2d.append(c2d)

        # Scatter
        ax2.scatter(nf_2d[:, 0], nf_2d[:, 1], s=6, alpha=0.15,
                    color=SERIES_COLORS[i])
        # Centroid
        ax2.scatter(*c2d, s=150, color=SERIES_COLORS[i], edgecolors='black',
                    linewidth=1.5, zorder=5, label=SERIES_LABELS[i])
        # 2-sigma ellipse
        confidence_ellipse(c2d, cov2d, ax2, n_std=2.0,
                          facecolor=SERIES_COLORS[i], alpha=0.15,
                          edgecolor=SERIES_COLORS[i], linewidth=2)

    # Draw trajectory arrows
    for j in range(len(centroids_2d) - 1):
        dx = centroids_2d[j+1][0] - centroids_2d[j][0]
        dy = centroids_2d[j+1][1] - centroids_2d[j][1]
        ax2.annotate('', xy=centroids_2d[j+1], xytext=centroids_2d[j],
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax2.set_title('NoFlow Cluster Drift Trajectory\n(centroids + 2σ ellipses, arrows = time direction)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(out_dir / 'noflow_drift_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir / 'noflow_drift_trajectory.png'}")

    # =================================================================
    # PLOT 3: Per-dimension centroid drift (raw latent, no PCA)
    # =================================================================
    series_present = [s for s in SERIES_ORDER if s in noflow_centroids]
    centroids_matrix = np.array([noflow_centroids[s] for s in series_present])
    latent_dim = centroids_matrix.shape[1]

    # Select top-8 most-drifting dimensions
    drift_range = centroids_matrix.max(axis=0) - centroids_matrix.min(axis=0)
    top_dims = np.argsort(drift_range)[::-1][:min(8, latent_dim)]

    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
    axes3 = axes3.flatten()
    x_pos = np.arange(len(series_present))

    for idx, dim in enumerate(top_dims):
        if idx >= len(axes3):
            break
        ax = axes3[idx]
        vals = centroids_matrix[:, dim]
        colors = [SERIES_COLORS[SERIES_ORDER.index(s)] for s in series_present]
        ax.bar(x_pos, vals, color=colors, edgecolor='white', width=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.split('.')[0] + '.' + s.split('.')[1]
                           for s in series_present], rotation=45, fontsize=8)
        ax.set_title(f'Dim {dim} (range={drift_range[dim]:.2f})', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    fig3.suptitle('NoFlow Centroid per Latent Dimension (top-8 drifting dims)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3.savefig(out_dir / 'noflow_centroid_per_dim.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir / 'noflow_centroid_per_dim.png'}")

    # =================================================================
    # PLOT 4: Drift magnitude over time (Mahalanobis from first series)
    # =================================================================
    ref_series = series_present[0]
    ref_centroid = noflow_centroids[ref_series]
    ref_cov = noflow_covs[ref_series]
    try:
        ref_cov_inv = np.linalg.inv(ref_cov + 1e-6 * np.eye(latent_dim))
    except np.linalg.LinAlgError:
        ref_cov_inv = np.linalg.pinv(ref_cov)

    mahal_distances = []
    eucl_distances = []
    for s in series_present:
        diff = noflow_centroids[s] - ref_centroid
        mahal = np.sqrt(diff @ ref_cov_inv @ diff)
        eucl = np.linalg.norm(diff)
        mahal_distances.append(mahal)
        eucl_distances.append(eucl)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    colors = [SERIES_COLORS[SERIES_ORDER.index(s)] for s in series_present]
    ax4a.bar(x_pos, mahal_distances, color=colors, edgecolor='white', width=0.6)
    ax4a.set_xticks(x_pos)
    ax4a.set_xticklabels(SERIES_LABELS[:len(series_present)], fontsize=8)
    ax4a.set_ylabel('Mahalanobis Distance', fontsize=12)
    ax4a.set_title(f'Drift from Reference ({ref_series})\nMahalanobis Distance', fontsize=12, fontweight='bold')
    ax4a.grid(axis='y', alpha=0.3)

    ax4b.bar(x_pos, eucl_distances, color=colors, edgecolor='white', width=0.6)
    ax4b.set_xticks(x_pos)
    ax4b.set_xticklabels(SERIES_LABELS[:len(series_present)], fontsize=8)
    ax4b.set_ylabel('Euclidean Distance', fontsize=12)
    ax4b.set_title(f'Drift from Reference ({ref_series})\nEuclidean Distance', fontsize=12, fontweight='bold')
    ax4b.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig4.savefig(out_dir / 'noflow_drift_magnitude.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir / 'noflow_drift_magnitude.png'}")

    # =================================================================
    # Summary stats to JSON
    # =================================================================
    summary = {}
    for i, s in enumerate(series_present):
        summary[s] = {
            'n_noflow': int((latents[s]['labels'] == 0).sum()),
            'n_flow': int((latents[s]['labels'] == 1).sum()),
            'centroid_pca': centroids_2d[i].tolist() if i < len(centroids_2d) else None,
            'mahalanobis_from_first': float(mahal_distances[i]),
            'euclidean_from_first': float(eucl_distances[i]),
        }
    with open(out_dir / 'drift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir / 'drift_summary.json'}")

    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
