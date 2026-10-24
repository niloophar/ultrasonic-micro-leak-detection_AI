#!/usr/bin/env python
"""
DeepSAD circle visualisation: center c, margin sphere, NoFlow vs Flow distribution.

For each panel the L2-normalised projection vectors z_proj are centred at the
Haus-A DeepSAD training center c and projected to 2D via PCA.  The margin
circle (radius m) marks the geometric boundary DeepSAD optimises: NoFlow
patches should cluster inside, Flow patches outside.

Four panels in a single 2x2 figure:
  A — Training set      (9 Haus-A train series, eval33 config)
  B — Haus-A held-out   (21.02, ha-2.05, ha-17.05)
  C — Haus-B before     (hb-26.04, hb-4.05 vs the Haus-A circle)
  D — Haus-B after      (same patches; circle shifted to hb- NoFlow centroid)

Research takeaway:
  A + B confirm DeepSAD geometry works for Haus A.
  C shows Haus B is geometrically displaced — all NoFlow outside the Haus-A circle.
  D shows that translating the circle to the hb- centroid suffices to enclose
  the Haus-B NoFlow cluster, without retraining the encoder.

Run on the GPU server from the project root:
    python scripts/visualize_deepsad_circles.py \\
        [--checkpoint PATH] [--scaler PATH] [--params PATH] \\
        [--data-path data] [--img-path images]

Default checkpoint:
    data/contrastive_g5_deepsad_transformer_ae_eval33/results/results/contrastive_model.pt
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contrastive.models import build_contrastive_model
from src.contrastive.train import (
    _load_splits_and_metadata,
    _ids_for_series,
    _prepare_patches,
    _extract_latents,
    _to_channels,
)
from src.common.utils import detect_n_stats


# eval33: 9-series DeepSAD Transformer AE
TRAIN_SERIES = [
    '10.01.2026', '24.01.2026', '7.02.2026',
    '7.03.2026', '15.03.2026', '5.04.2026',
    '11.04.2026', '18.04.2026', 'ha-25.04.2026',
]
TEST_SERIES_HA = ['21.02.2026', 'ha-2.05.2026', 'ha-17.05.2026']
HB_SERIES = ['hb-26.04.2026', 'hb-4.05.2026']

DEFAULT_CKPT = (
    'data/contrastive_g5_deepsad_transformer_ae_eval33'
    '/results/results/contrastive_model.pt'
)
DEFAULT_SCALER = (
    'data/contrastive_g5_deepsad_transformer_ae_eval33'
    '/results/results/input_scaler.pkl'
)
DEFAULT_PARAMS = (
    'data/contrastive_g5_deepsad_transformer_ae_eval33'
    '/results/results/optuna_best_params.json'
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get('hp', {})
    model = build_contrastive_model(
        encoder_type=hp.get('encoder_type', 'autoencoder'),
        encoder_arch=hp.get('encoder_arch', 'transformer'),
        in_channels=ckpt.get('n_in_channels', 2),
        input_height=ckpt.get('input_height', 9),
        input_width=ckpt.get('input_width', 64),
        latent_dim=hp.get('latent_dim', 128),
        projection_dim=hp.get('projection_dim', 64),
        channels=hp.get('cnn_channels', [32, 64, 128]),
        kernel_size=hp.get('cnn_kernel_size', 3),
        dropout=hp.get('dropout', 0.3),
        d_model=hp.get('d_model', 128),
        num_heads=hp.get('num_heads', 2),
        num_layers=hp.get('num_layers', 6),
        ff_dim=hp.get('ff_dim', 256),
        transformer_dropout=hp.get('transformer_dropout', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded model  {ckpt_path}  (best_epoch={ckpt.get('best_epoch', '?')})")
    return model, hp


def _apply_saved_scaler(patches: np.ndarray, scaler) -> np.ndarray:
    """Apply the NoFlow-only StandardScaler fitted during training.

    Scaler was fitted on (time_step, feature) pairs individually, so we need
    the same transpose-reshape pattern as _scale_patches in train.py.
    """
    if len(patches) == 0:
        return patches
    n, f, w = patches.shape
    flat = patches.transpose(0, 2, 1).reshape(-1, f)
    return scaler.transform(flat).reshape(n, w, f).transpose(0, 2, 1).astype(np.float32)


def _load_group(
    series_names, splits_data, metadata_by_id, prepared_base,
    selected_features, scaler, n_stats, dual,
):
    """Load patches for a list of series, apply scaler and channel reshape."""
    ids = _ids_for_series(splits_data, metadata_by_id, series_names)
    if not ids:
        print(f"  [warn] no segments found for {series_names}")
        return np.empty((0, 2, 9, 64), dtype=np.float32), np.empty(0, dtype=int), []

    patches, labels, series, _ = _prepare_patches(
        ids, prepared_base, metadata_by_id, selected_features
    )
    if len(patches) == 0:
        print(f"  [warn] no patches loaded for {series_names}")
        return np.empty((0, 2, 9, 64), dtype=np.float32), np.empty(0, dtype=int), []

    patches = _apply_saved_scaler(patches, scaler)
    f, w = patches.shape[1], patches.shape[2]
    patches_ch, _ = _to_channels(patches, f, w, n_stats, dual)
    return patches_ch, np.array(labels), series


# ---------------------------------------------------------------------------
# DeepSAD center + circle drawing helpers
# ---------------------------------------------------------------------------

def _compute_center(z_proj_noflow: np.ndarray) -> np.ndarray:
    """L2-normalise the mean of NoFlow z_proj — matches _init_deepsad_center."""
    mean = z_proj_noflow.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-12)


def _draw_circle(ax, center_2d, radius, color='black', linestyle='--',
                 linewidth=2.0, alpha=1.0, label=None, zorder=3):
    c = Circle(center_2d, radius, fill=False,
               edgecolor=color, linestyle=linestyle,
               linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)
    ax.add_patch(c)
    return c


def _scatter_panel(ax, xy_nf, xy_fl, label_nf='NoFlow', label_fl='Flow',
                   color_nf='#1f77b4', color_fl='#d62728',
                   marker_nf='o', marker_fl='o', s=12, alpha=0.45, zorder=2):
    if len(xy_nf) > 0:
        ax.scatter(xy_nf[:, 0], xy_nf[:, 1], s=s, c=color_nf,
                   marker=marker_nf, alpha=alpha, label=label_nf, zorder=zorder)
    if len(xy_fl) > 0:
        ax.scatter(xy_fl[:, 0], xy_fl[:, 1], s=s, c=color_fl,
                   marker=marker_fl, alpha=alpha, label=label_fl, zorder=zorder)


def _annotate_distances(ax, z_proj, labels, center, margin, fontsize=8):
    """Print mean dist from center for NoFlow and Flow in a text box."""
    d = np.linalg.norm(z_proj - center, axis=1)
    nf_mask = labels == 0
    fl_mask = labels == 1
    lines = []
    if nf_mask.any():
        lines.append(f'NoFlow d̄ = {d[nf_mask].mean():.3f}  (m={margin:.3f})')
    if fl_mask.any():
        lines.append(f'Flow   d̄ = {d[fl_mask].mean():.3f}')
    text = '\n'.join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def _style_ax(ax, title, xlim, ylim):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axhline(0, color='grey', linewidth=0.4, linestyle=':', zorder=1)
    ax.axvline(0, color='grey', linewidth=0.4, linestyle=':', zorder=1)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT)
    parser.add_argument('--scaler', default=DEFAULT_SCALER)
    parser.add_argument('--params', default=DEFAULT_PARAMS,
                        help='optuna_best_params.json (for deepsad_margin)')
    parser.add_argument('--data-path', default='data')
    parser.add_argument('--data-root', default='./prepared')
    parser.add_argument('--img-path', default='images')
    args = parser.parse_args()

    import os
    if os.getenv('DOCKER_ENV'):
        root = Path(os.getenv('CONTAINER_APP_PATH', '/app'))
        data_path = Path(os.getenv('CONTAINER_DATA_PATH', '/app/data'))
        img_path = Path(os.getenv('CONTAINER_IMG_PATH', '/app/images'))
    else:
        root = Path(__file__).resolve().parent.parent
        data_path = root / args.data_path
        img_path = root / args.img_path

    ckpt_path = root / args.checkpoint
    scaler_path = root / args.scaler
    params_path = root / args.params

    out_dir = img_path / 'contrastive' / 'deepsad_circles'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load model + scaler ---
    model, hp = _load_model(ckpt_path, device)

    with open(scaler_path, 'rb') as f:
        scaler_bundle = pickle.load(f)
    scaler = scaler_bundle['scaler']
    feat_names = scaler_bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    dual = (scaler_bundle.get('cnn_channel_mode', 'dual') == 'dual')
    selected_features = list(feat_names) if feat_names else None

    with open(params_path) as f:
        params = json.load(f)
    margin = float(params.get('params', params).get('deepsad_margin', 1.376))
    print(f'  DeepSAD margin m = {margin:.4f}')

    # --- Load splits + metadata ---
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(
        data_path, args.data_root
    )

    # --- Load patches for all three groups ---
    print('\nLoading patches...')
    tr_ch, tr_lab, tr_ser = _load_group(
        TRAIN_SERIES, splits_data, metadata_by_id, prepared_base,
        selected_features, scaler, n_stats, dual
    )
    te_ch, te_lab, te_ser = _load_group(
        TEST_SERIES_HA, splits_data, metadata_by_id, prepared_base,
        selected_features, scaler, n_stats, dual
    )
    hb_ch, hb_lab, hb_ser = _load_group(
        HB_SERIES, splits_data, metadata_by_id, prepared_base,
        selected_features, scaler, n_stats, dual
    )
    print(f'  Train:  {len(tr_ch)} patches   Test-HA: {len(te_ch)}   hb-: {len(hb_ch)}')

    # --- Extract z_proj ---
    print('\nExtracting z_proj (projection head output)...')
    tr_zp = _extract_latents(model, tr_ch, device, use_proj=True)
    te_zp = _extract_latents(model, te_ch, device, use_proj=True) if len(te_ch) > 0 else np.empty((0, tr_zp.shape[1]))
    hb_zp = _extract_latents(model, hb_ch, device, use_proj=True) if len(hb_ch) > 0 else np.empty((0, tr_zp.shape[1]))
    print(f'  z_proj dim = {tr_zp.shape[1]}')

    # --- Compute Haus-A DeepSAD center c_ha ---
    tr_nf_mask = tr_lab == 0
    c_ha = _compute_center(tr_zp[tr_nf_mask])
    print(f'  Center c_ha norm = {np.linalg.norm(c_ha):.4f}  '
          f'(from {tr_nf_mask.sum()} train NoFlow patches)')

    # --- Compute Haus-B adapted center c_hb (all hb- are NoFlow) ---
    hb_available = len(hb_zp) > 0
    if hb_available:
        c_hb = _compute_center(hb_zp)
        hb_dists_from_hb_center = np.linalg.norm(hb_zp - c_hb, axis=1)
        margin_hb = float(np.percentile(hb_dists_from_hb_center, 90))
        print(f'  Center c_hb norm = {np.linalg.norm(c_hb):.4f}  '
              f'margin_hb (p90) = {margin_hb:.4f}')
        dist_ha_to_hb = np.linalg.norm(c_ha - c_hb)
        print(f'  ||c_ha - c_hb|| = {dist_ha_to_hb:.4f}  '
              f'({dist_ha_to_hb / margin:.1f}× the Haus-A margin)')

    # --- PCA in z_proj space centred at c_ha ---
    # Collect all z_proj, centre by c_ha, fit PCA
    all_zp = [tr_zp, te_zp]
    if hb_available:
        all_zp.append(hb_zp)
    all_zp_cat = np.concatenate([z for z in all_zp if len(z) > 0], axis=0)

    all_centred = all_zp_cat - c_ha
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_centred)
    var_exp = pca.explained_variance_ratio_ * 100
    print(f'\n  PCA explained variance: PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%')

    def _proj(z):
        if len(z) == 0:
            return np.empty((0, 2))
        return pca.transform(z - c_ha)

    tr_2d = _proj(tr_zp)
    te_2d = _proj(te_zp)
    hb_2d = _proj(hb_zp)

    # Origin = projected c_ha  (always 0,0 by construction)
    origin = np.array([0.0, 0.0])

    # Projected Haus-B center
    if hb_available:
        c_hb_2d = pca.transform((c_hb - c_ha).reshape(1, -1))[0]

    # --- Auto-scale axes to cover all data + circles ---
    all_2d = np.concatenate([tr_2d, te_2d, hb_2d] if hb_available else [tr_2d, te_2d], axis=0)
    pad = max(margin * 1.5, np.abs(all_2d).max() * 1.1)
    lim = (-pad, pad)

    # ---------------------------------------------------------------------------
    # Figure: 2×2 panels
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(
        f'DeepSAD Geometry in z_proj Space (eval33, m={margin:.3f})\n'
        f'PCA PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  '
        f'(centred at Haus-A training NoFlow center c)',
        fontsize=12,
    )

    # ---- Panel A: Training set ----
    ax = axes[0, 0]
    xy_nf = tr_2d[tr_lab == 0]
    xy_fl = tr_2d[tr_lab == 1]
    _scatter_panel(ax, xy_nf, xy_fl)
    _draw_circle(ax, origin, margin, color='#2ca02c', linewidth=2.5,
                 label=f'Haus-A margin (m={margin:.3f})')
    ax.scatter(*origin, marker='+', s=120, c='#2ca02c', linewidths=2, zorder=5,
               label='Center c (HA)')
    _annotate_distances(ax, tr_zp, tr_lab, c_ha, margin)
    _style_ax(ax, 'A — Training set (9 Haus-A series)', lim, lim)

    # ---- Panel B: Haus-A held-out test ----
    ax = axes[0, 1]
    te_nf_mask = te_lab == 0
    te_fl_mask = te_lab == 1
    _scatter_panel(ax, te_2d[te_nf_mask], te_2d[te_fl_mask])
    _draw_circle(ax, origin, margin, color='#2ca02c', linewidth=2.5,
                 label=f'Haus-A margin (m={margin:.3f})')
    ax.scatter(*origin, marker='+', s=120, c='#2ca02c', linewidths=2, zorder=5,
               label='Center c (HA)')
    _annotate_distances(ax, te_zp, te_lab, c_ha, margin)
    _style_ax(ax, 'B — Haus-A held-out (21.02, ha-2.05, ha-17.05)', lim, lim)

    # ---- Panel C: Haus-B before adaptation ----
    ax = axes[1, 0]
    if hb_available:
        # All hb- patches here are NoFlow (no leakage in hb-26.04/hb-4.05)
        ax.scatter(hb_2d[:, 0], hb_2d[:, 1], s=14, c='#ff7f0e',
                   marker='^', alpha=0.55, label='hb- NoFlow', zorder=2)
        _draw_circle(ax, origin, margin, color='#2ca02c', linewidth=2.5,
                     label=f'Haus-A margin (m={margin:.3f})')
        ax.scatter(*origin, marker='+', s=120, c='#2ca02c', linewidths=2, zorder=5,
                   label='Center c (HA)')
        ax.scatter(*c_hb_2d, marker='x', s=120, c='#ff7f0e', linewidths=2, zorder=5,
                   label='hb- centroid')
        d_hb_from_ha = np.linalg.norm(hb_zp - c_ha, axis=1)
        ax.text(0.02, 0.98,
                f'hb- NoFlow d̄ from c_HA = {d_hb_from_ha.mean():.3f}\n'
                f'({d_hb_from_ha.mean()/margin:.1f}× margin)\n'
                f'n = {len(hb_zp)}',
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No hb- patches found.\nRun preparation for hb- series.',
                transform=ax.transAxes, ha='center', va='center')
    _style_ax(ax, 'C — Haus-B before adaptation', lim, lim)

    # ---- Panel D: Haus-B after adaptation (adapted circle) ----
    ax = axes[1, 1]
    if hb_available:
        ax.scatter(hb_2d[:, 0], hb_2d[:, 1], s=14, c='#ff7f0e',
                   marker='^', alpha=0.55, label='hb- NoFlow', zorder=2)
        # Haus-A circle (faded reference)
        _draw_circle(ax, origin, margin, color='#2ca02c', linewidth=1.2,
                     alpha=0.35, linestyle=':', label=f'Haus-A margin (ref)')
        ax.scatter(*origin, marker='+', s=80, c='#2ca02c', linewidths=1.5,
                   alpha=0.4, zorder=5)
        # Adapted circle centred at hb- centroid
        _draw_circle(ax, c_hb_2d, margin_hb, color='#ff7f0e', linewidth=2.5,
                     label=f'Adapted margin (p90={margin_hb:.3f})')
        ax.scatter(*c_hb_2d, marker='x', s=120, c='#ff7f0e', linewidths=2,
                   zorder=5, label='Adapted center c_hb')
        d_hb_from_hb = np.linalg.norm(hb_zp - c_hb, axis=1)
        pct_inside = float((d_hb_from_hb < margin_hb).mean()) * 100
        ax.text(0.02, 0.98,
                f'hb- NoFlow d̄ from c_hb = {d_hb_from_hb.mean():.4f}\n'
                f'p90 = {margin_hb:.4f}  (adapted margin)\n'
                f'{pct_inside:.0f}% inside adapted circle',
                transform=ax.transAxes, fontsize=8, va='top', ha='left',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No hb- patches found.', transform=ax.transAxes,
                ha='center', va='center')
    _style_ax(ax, 'D — Haus-B after adaptation (circle translated)', lim, lim)

    plt.tight_layout()
    out_path = out_dir / 'deepsad_circles_eval33.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved → {out_path}')

    # --- Also save one high-res panel per figure for study-project inclusion ---
    for panel_idx, (panel_letter, panel_title) in enumerate([
        ('A', 'training'), ('B', 'test_ha'), ('C', 'hb_before'), ('D', 'hb_after'),
    ]):
        fig_s, ax_s = plt.subplots(1, 1, figsize=(5.5, 5.0))
        row, col = divmod(panel_idx, 2)
        # Re-draw the same panel
        if panel_letter == 'A':
            _scatter_panel(ax_s, tr_2d[tr_lab == 0], tr_2d[tr_lab == 1])
            _draw_circle(ax_s, origin, margin, color='#2ca02c', linewidth=2.5,
                         label=f'Margin (m={margin:.3f})')
            ax_s.scatter(*origin, marker='+', s=120, c='#2ca02c', linewidths=2, zorder=5, label='c')
            _annotate_distances(ax_s, tr_zp, tr_lab, c_ha, margin)
            _style_ax(ax_s, f'{panel_letter} — Training (Haus A, 9 series)', lim, lim)
        elif panel_letter == 'B':
            _scatter_panel(ax_s, te_2d[te_lab == 0], te_2d[te_lab == 1])
            _draw_circle(ax_s, origin, margin, color='#2ca02c', linewidth=2.5,
                         label=f'Margin (m={margin:.3f})')
            ax_s.scatter(*origin, marker='+', s=120, c='#2ca02c', linewidths=2, zorder=5, label='c')
            _annotate_distances(ax_s, te_zp, te_lab, c_ha, margin)
            _style_ax(ax_s, 'B — Eval held-out (21.02 / ha-2.05 / ha-17.05)', lim, lim)
        elif panel_letter == 'C' and hb_available:
            ax_s.scatter(hb_2d[:, 0], hb_2d[:, 1], s=18, c='#ff7f0e',
                         marker='^', alpha=0.6, label='hb- NoFlow', zorder=2)
            _draw_circle(ax_s, origin, margin, color='#2ca02c', linewidth=2.5,
                         label=f'Haus-A margin (m={margin:.3f})')
            ax_s.scatter(*origin, marker='+', s=120, c='#2ca02c', linewidths=2, zorder=5, label='c (HA)')
            ax_s.scatter(*c_hb_2d, marker='x', s=120, c='#ff7f0e', linewidths=2, zorder=5, label='c_hb')
            _style_ax(ax_s, 'C — Haus B before adaptation', lim, lim)
        elif panel_letter == 'D' and hb_available:
            ax_s.scatter(hb_2d[:, 0], hb_2d[:, 1], s=18, c='#ff7f0e',
                         marker='^', alpha=0.6, label='hb- NoFlow', zorder=2)
            _draw_circle(ax_s, origin, margin, color='#2ca02c', linewidth=1.2,
                         alpha=0.3, linestyle=':', label='Haus-A margin (ref)')
            _draw_circle(ax_s, c_hb_2d, margin_hb, color='#ff7f0e', linewidth=2.5,
                         label=f'Adapted (p90={margin_hb:.3f})')
            ax_s.scatter(*c_hb_2d, marker='x', s=120, c='#ff7f0e', linewidths=2, zorder=5, label='c_hb')
            _style_ax(ax_s, 'D — Haus B after adaptation', lim, lim)
        else:
            plt.close(fig_s)
            continue

        fig_s.tight_layout()
        out_single = out_dir / f'deepsad_circle_{panel_letter}_{panel_title}.png'
        fig_s.savefig(out_single, dpi=150, bbox_inches='tight')
        plt.close(fig_s)
        print(f'  Saved → {out_single}')

    print('\nDone.')


if __name__ == '__main__':
    main()
