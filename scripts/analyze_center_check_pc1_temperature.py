#!/usr/bin/env python
"""
Follow-up to the Part 1 pooled two-house DeepSAD center diagnostic
(scripts/analyze_deepsad_center_check.py): is the PC1 drift seen there
explained by temperature?

Same methodology as the established scripts/analyze_drift_temperature.py
(per-SERIES NoFlow centroids in latent space, PCA fit on those centroids,
Pearson+Spearman correlation against per-series mean temp_pipe/temp_amb) —
applied here to the pooled two-house DeepSAD Transformer checkpoint instead
of the earlier CNN encoder, and broken out per INDIVIDUAL series (not just
pooled per-house) so the correlation has enough points to mean something.

Run inside Docker (needs data/prepared/ + data/processed/ + a trained checkpoint):
    docker compose run --rm kild python scripts/analyze_center_check_pc1_temperature.py \
        --output-root ./contrastive_g5_deepsad_transformer_ae_hb_diversity

Outputs:
  - data/results/haus_b_adaptation/center_check_pc1_temperature.json
  - images/contrastive/deepsad_center_check/pc1_vs_temperature.png
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
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contrastive.train import (
    _load_splits_and_metadata,
    _ids_for_series,
    _prepare_patches,
    _extract_latents,
    _compute_per_series_regress_target,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'data' / 'results' / 'haus_b_adaptation'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = ROOT / 'images' / 'contrastive' / 'deepsad_center_check'
IMG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HAUS_A = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026', '7.03.2026',
    '15.03.2026', '5.04.2026', '11.04.2026', '18.04.2026', 'ha-25.04.2026',
    'ha-2.05.2026', 'ha-9.05.2026', 'ha-17.05.2026', 'ha-24.05.2026',
    'ha-6.06.2026', 'ha-2.06.2026', 'ha-20.06.2026',
]
DEFAULT_HAUS_B = ['hb-17.05.2026', 'hb-25.05.2026']

C_HA = '#2166ac'
C_HB = '#c0392b'


def main():
    import pickle
    from src.common.utils import detect_n_stats
    from src.contrastive.models import build_contrastive_model
    from src.contrastive.train import _to_channels
    from src.contrastive.main import FEATURE_GROUPS

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=str, default='./contrastive_g5_deepsad_transformer_ae_hb_diversity')
    parser.add_argument('--data-root', type=str, default='./prepared')
    parser.add_argument('--haus-a-series', nargs='+', default=DEFAULT_HAUS_A)
    parser.add_argument('--haus-b-series', nargs='+', default=DEFAULT_HAUS_B)
    parser.add_argument('--feature-group', type=str, default='g5', choices=list(FEATURE_GROUPS.keys()))
    parser.add_argument('--temp-target', type=str, default='temp_pipe', choices=['temp_pipe', 'temp_amb', 'temp_diff'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = ROOT / 'data' / args.output_root.lstrip('./') / 'results'
    ckpt = torch.load(results_dir / 'contrastive_model.pt', map_location=device, weights_only=False)
    hp = ckpt['hp']
    model = build_contrastive_model(
        encoder_type=hp['encoder_type'], encoder_arch=hp.get('encoder_arch', 'cnn'),
        in_channels=ckpt['n_in_channels'], input_height=ckpt['input_height'], input_width=ckpt['input_width'],
        latent_dim=hp['latent_dim'], projection_dim=hp['projection_dim'],
        channels=hp.get('cnn_channels', [32, 64, 128]), kernel_size=hp.get('cnn_kernel_size', 3),
        dropout=hp.get('dropout', 0.3), d_model=hp.get('d_model', 128), num_heads=hp.get('num_heads', 4),
        num_layers=hp.get('num_layers', 4), ff_dim=hp.get('ff_dim', 256),
        transformer_dropout=hp.get('transformer_dropout', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {results_dir} (best_epoch={ckpt.get('best_epoch', '?')})")

    with open(results_dir / 'input_scaler.pkl', 'rb') as f:
        scaler_bundle = pickle.load(f)
    saved_scaler = scaler_bundle['scaler']
    feat_names_scaler = scaler_bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names_scaler) if feat_names_scaler else 1
    dual = (scaler_bundle.get('cnn_channel_mode', 'dual') == 'dual')

    def scale_and_channel(patches: np.ndarray) -> np.ndarray:
        if saved_scaler is not None:
            n_, f_, w_ = patches.shape
            flat = patches.transpose(0, 2, 1).reshape(-1, f_)
            patches = saved_scaler.transform(flat).reshape(n_, w_, f_).transpose(0, 2, 1).astype(np.float32)
        channels, _ = _to_channels(patches, patches.shape[1], patches.shape[2], n_stats, dual)
        return channels

    data_path = ROOT / 'data'
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, args.data_root)
    selected = FEATURE_GROUPS[args.feature_group]

    all_series = list(args.haus_a_series) + list(args.haus_b_series)
    temp_map = _compute_per_series_regress_target(data_path, all_series, args.temp_target)

    centroids = []
    series_labels = []
    house_labels = []
    temps = []
    n_patches_per_series = {}

    for house, series_list in (('Haus A', args.haus_a_series), ('Haus B', args.haus_b_series)):
        for s in series_list:
            ids = _ids_for_series(splits_data, metadata_by_id, [s])
            if not ids:
                print(f"  [skip] {s} — no segments found")
                continue
            patches, labels, _, _ = _prepare_patches(ids, prepared_base, metadata_by_id, selected)
            noflow = patches[np.asarray(labels) == 0]
            if len(noflow) == 0:
                print(f"  [skip] {s} — no NoFlow patches")
                continue
            noflow_c = scale_and_channel(noflow)
            z = _extract_latents(model, noflow_c, device, use_proj=True)
            centroids.append(z.mean(axis=0))
            series_labels.append(s)
            house_labels.append(house)
            temps.append(temp_map.get(s, float('nan')))
            n_patches_per_series[s] = int(len(noflow))
            print(f"  {house:8s} {s:16s} n_noflow={len(noflow):5d} {args.temp_target}={temp_map.get(s, float('nan')):.3f}")

    centroids = np.array(centroids)
    temps = np.array(temps, dtype=np.float64)

    pca = PCA(n_components=2, random_state=42).fit(centroids)
    centroids_pc = pca.transform(centroids)
    pc1 = centroids_pc[:, 0]
    pc2 = centroids_pc[:, 1]

    valid = ~np.isnan(temps)
    pear_pc1, pear_pc1_p = pearsonr(temps[valid], pc1[valid])
    spear_pc1, spear_pc1_p = spearmanr(temps[valid], pc1[valid])
    pear_pc2, pear_pc2_p = pearsonr(temps[valid], pc2[valid])
    spear_pc2, spear_pc2_p = spearmanr(temps[valid], pc2[valid])

    summary = {
        'temp_target': args.temp_target,
        'n_series': len(series_labels),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'pc1_vs_temp_pearson_r': float(pear_pc1), 'pc1_vs_temp_pearson_p': float(pear_pc1_p),
        'pc1_vs_temp_spearman_r': float(spear_pc1), 'pc1_vs_temp_spearman_p': float(spear_pc1_p),
        'pc2_vs_temp_pearson_r': float(pear_pc2), 'pc2_vs_temp_pearson_p': float(pear_pc2_p),
        'pc2_vs_temp_spearman_r': float(spear_pc2), 'pc2_vs_temp_spearman_p': float(spear_pc2_p),
        'per_series': [
            {'series': s, 'house': h, 'temp': float(t), 'pc1': float(p1), 'pc2': float(p2),
             'n_noflow_patches': n_patches_per_series[s]}
            for s, h, t, p1, p2 in zip(series_labels, house_labels, temps, pc1, pc2)
        ],
    }
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_series'}, indent=2))

    out_path = OUT_DIR / 'center_check_pc1_temperature.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved -> {out_path}')

    # --- Plot: PC1 vs temperature, coloured by house ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, pc_vals, pc_name, r_p, r_s in (
        (axes[0], pc1, f'Centroid PC1 ({pca.explained_variance_ratio_[0]:.0%} var)', pear_pc1, spear_pc1),
        (axes[1], pc2, f'Centroid PC2 ({pca.explained_variance_ratio_[1]:.0%} var)', pear_pc2, spear_pc2),
    ):
        for h, color in (('Haus A', C_HA), ('Haus B', C_HB)):
            mask = np.array([hl == h for hl in house_labels])
            ax.scatter(temps[mask], pc_vals[mask], s=90, color=color, label=h,
                       edgecolor='white', linewidth=0.8, zorder=3)
        for s, t, p in zip(series_labels, temps, pc_vals):
            ax.annotate(s.replace('.2026', ''), (t, p), fontsize=6.5, color='#555555',
                        xytext=(4, 3), textcoords='offset points')
        ax.set_xlabel(f'Mean {args.temp_target} [°C] (per series, NoFlow)')
        ax.set_ylabel(pc_name)
        ax.set_title(f'Pearson r={r_p:.2f}, Spearman ρ={r_s:.2f}')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Does {args.temp_target} explain the pooled two-house latent drift?', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = IMG_DIR / 'pc1_vs_temperature.png'
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f'Saved -> {plot_path}')


if __name__ == '__main__':
    main()
