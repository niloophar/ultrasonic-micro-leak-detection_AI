#!/usr/bin/env python
"""
Raw (pre-encoder) feature distribution comparison: Haus A NoFlow vs. Haus B
NoFlow, on the exact G5 features the encoder actually consumes.

Motivation (supervisor question, 2026-07-16): every Haus A/B comparison done
so far (PCA drift plots, centroid trajectories) was in the ENCODER'S LATENT
space, never on the raw input features. The production encoder's input
StandardScaler is fit on Haus A NoFlow only (Haus B is never in
encoder_train_series) — so part of what's been called "Haus B is a separate
drift axis" could partly be a normalization-mismatch artefact rather than a
genuine representation-level site difference. This script checks that
directly.

Pipeline:
  1. Load raw (unscaled) G5 patches for Haus A NoFlow and Haus B NoFlow via
     the same _prepare_patches path the encoder itself uses.
  2. Report per-feature raw mean/std for each house.
  3. Fit a StandardScaler on Haus A NoFlow only (matches the production
     encoder's SCALER_A convention) and transform both houses through it —
     Haus A should land near mean=0/std=1 by construction; how far Haus B
     deviates quantifies the scaler mismatch.
  4. Per-feature comparison plot (raw + Haus-A-scaled) + summary JSON.

Run inside Docker (needs data/prepared/):
    docker compose run --rm kild python scripts/analyze_haus_ab_feature_distributions.py

Outputs:
  - images/contrastive/haus_ab_features/feature_distribution_comparison.png
  - data/results/haus_b_adaptation/haus_ab_feature_distribution_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contrastive.train import (
    _load_splits_and_metadata,
    _ids_for_series,
    _prepare_patches,
)
from src.contrastive.main import FEATURE_GROUPS

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / 'images' / 'contrastive' / 'haus_ab_features'
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / 'data' / 'results' / 'haus_b_adaptation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAUS_A_SERIES = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026', '7.03.2026',
    '15.03.2026', '5.04.2026', '11.04.2026', '18.04.2026', 'ha-25.04.2026',
    'ha-2.05.2026', 'ha-9.05.2026', 'ha-17.05.2026', 'ha-24.05.2026',
    'ha-6.06.2026', 'ha-2.06.2026', 'ha-20.06.2026',
]
HAUS_B_SERIES = [
    'hb-26.04.2026', 'hb-4.05.2026', 'hb-10.05.2026', 'hb-17.05.2026',
    'hb-25.05.2026', 'hb-31.05.2026', 'hb-7.06.2026', 'hb-14.06.2026',
]

C_HA = '#2166ac'
C_HB = '#c0392b'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./prepared')
    args = parser.parse_args()

    data_path = ROOT / 'data'
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, args.data_root)

    # G5 selected features — same 18 norm_interval_up/down features the
    # production encoder actually consumes (NOT all 282 raw+derived columns).
    selected = FEATURE_GROUPS['g5']

    ha_ids = _ids_for_series(splits_data, metadata_by_id, HAUS_A_SERIES)
    hb_ids = _ids_for_series(splits_data, metadata_by_id, HAUS_B_SERIES)
    print(f"Haus A segments: {len(ha_ids)} | Haus B segments: {len(hb_ids)}")

    ha_patches, ha_labels, ha_series, feat_names = _prepare_patches(ha_ids, prepared_base, metadata_by_id, selected)
    hb_patches, hb_labels, hb_series, _ = _prepare_patches(hb_ids, prepared_base, metadata_by_id, selected)

    ha_noflow = ha_patches[np.asarray(ha_labels) == 0]  # (N, F, W)
    hb_noflow = hb_patches[np.asarray(hb_labels) == 0]
    print(f"Haus A NoFlow patches: {ha_noflow.shape} | Haus B NoFlow patches: {hb_noflow.shape}")

    n_feat = ha_noflow.shape[1]
    feat_names = feat_names or [f'feature_{i}' for i in range(n_feat)]

    # Flatten (N, F, W) -> (N*W, F): every raw point, not just per-patch means.
    def flatten(patches):
        n, f, w = patches.shape
        return patches.transpose(0, 2, 1).reshape(-1, f)

    ha_flat = flatten(ha_noflow)
    hb_flat = flatten(hb_noflow)

    # --- Raw (unscaled) per-feature stats ---
    summary = {'n_features': n_feat, 'feature_names': feat_names, 'per_feature': []}
    for i, name in enumerate(feat_names):
        ha_col, hb_col = ha_flat[:, i], hb_flat[:, i]
        ha_mean, ha_std = float(ha_col.mean()), float(ha_col.std())
        hb_mean, hb_std = float(hb_col.mean()), float(hb_col.std())
        mean_gap_in_ha_std = (hb_mean - ha_mean) / ha_std if ha_std > 0 else float('nan')
        summary['per_feature'].append({
            'feature': name,
            'haus_a_mean': ha_mean, 'haus_a_std': ha_std,
            'haus_b_mean': hb_mean, 'haus_b_std': hb_std,
            'mean_gap_in_haus_a_std': mean_gap_in_ha_std,
        })

    # --- Scaler-mismatch test: fit StandardScaler on Haus A NoFlow only,
    #     transform both houses through it (matches production SCALER_A). ---
    scaler = StandardScaler().fit(ha_flat)
    ha_scaled = scaler.transform(ha_flat)
    hb_scaled = scaler.transform(hb_flat)
    summary['scaler_mismatch'] = {
        'haus_a_scaled_mean_overall': float(ha_scaled.mean()),
        'haus_a_scaled_std_overall': float(ha_scaled.std()),
        'haus_b_scaled_mean_overall': float(hb_scaled.mean()),
        'haus_b_scaled_std_overall': float(hb_scaled.std()),
        'note': ('Haus A should land near mean=0/std=1 by construction (the scaler was fit on it). '
                 'Haus B deviating far from mean=0/std=1 here means part of the observed latent-space '
                 '"Haus B drift" is attributable to the scaler never having seen Haus B, not necessarily '
                 'a genuine site-identity signal the encoder itself introduces.'),
    }

    summary_path = OUT_DIR / 'haus_ab_feature_distribution_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved -> {summary_path}')
    print(f"Scaler mismatch: Haus B scaled mean={summary['scaler_mismatch']['haus_b_scaled_mean_overall']:.3f}, "
          f"std={summary['scaler_mismatch']['haus_b_scaled_std_overall']:.3f} "
          f"(Haus A is ~0/~1 by construction)")

    # --- Plot: per-feature raw distributions (top) + mean gap in Haus-A-sigma units (bottom) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    gaps = [p['mean_gap_in_haus_a_std'] for p in summary['per_feature']]
    # Whisker = Haus B's own std, expressed in the same Haus-A-std units as the
    # bar itself — shows spread mismatch alongside the mean shift (ties into
    # the scaler-mismatch finding: Haus B lands ~7.5x wider than Haus A).
    hb_std_whiskers = [p['haus_b_std'] / p['haus_a_std'] if p['haus_a_std'] > 0 else 0.0
                        for p in summary['per_feature']]
    colors = ['#c0392b' if abs(g) > 1 else '#2e7d32' for g in gaps]
    axes[0].bar(range(n_feat), gaps, yerr=hb_std_whiskers, capsize=3,
                color=colors, error_kw={'ecolor': '#333333', 'elinewidth': 1.0})
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].axhline(1, color='gray', lw=0.8, ls=':')
    axes[0].axhline(-1, color='gray', lw=0.8, ls=':')
    axes[0].set_xticks(range(n_feat))
    axes[0].set_xticklabels(feat_names, rotation=60, ha='right', fontsize=8)
    axes[0].set_ylabel('Haus B mean − Haus A mean\n(in Haus-A std units)')
    axes[0].set_title('Per-Feature Raw Mean Gap, Haus A vs. Haus B NoFlow\n(whiskers = ±1 Haus B std, same units)')

    # A representative feature's overlaid histogram (largest gap) for a concrete visual.
    worst_idx = int(np.nanargmax(np.abs(gaps)))
    worst_name = feat_names[worst_idx]
    ha_worst, hb_worst = ha_flat[:, worst_idx], hb_flat[:, worst_idx]

    # Diagnose the previous run's large left-side white space before deciding
    # how to fix it: print full range + percentiles for both houses.
    for label, arr in (('Haus A', ha_worst), ('Haus B', hb_worst)):
        p01, p50, p99 = np.percentile(arr, [1, 50, 99])
        print(f"  [{label}] {worst_name}: min={arr.min():.6f} p01={p01:.6f} "
              f"median={p50:.6f} p99={p99:.6f} max={arr.max():.6f} n={len(arr)}")
    combined_min = min(ha_worst.min(), hb_worst.min())
    combined_p01 = min(np.percentile(ha_worst, 1), np.percentile(hb_worst, 1))
    print(f"  Combined min={combined_min:.6f} vs combined p01={combined_p01:.6f} — "
          f"{'a small number of low outliers are stretching the x-axis' if combined_min < combined_p01 - 0.001 else 'range looks consistent, no obvious outliers'}")

    bin_edges = np.histogram_bin_edges(np.concatenate([ha_worst, hb_worst]), bins=80)
    axes[1].hist(ha_worst, bins=bin_edges, alpha=0.5, label='Haus A NoFlow', color=C_HA)
    axes[1].hist(hb_worst, bins=bin_edges, alpha=0.5, label='Haus B NoFlow', color=C_HB)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Count (log scale)')
    axes[1].set_title(f'Largest-gap feature: {worst_name}')
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out_path = IMG_DIR / 'feature_distribution_comparison.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
