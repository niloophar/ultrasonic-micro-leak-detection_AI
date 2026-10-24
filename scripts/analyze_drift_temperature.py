#!/usr/bin/env python
"""
Drift-Temperature Analysis for the Contrastive Encoder.

Answers: is temperature (pipe, ambient, or pipe-ambient diff) a suitable
axis for modelling the NoFlow-cluster drift in latent space? If yes, a
parametric model centroid = f(temperature) can replace blind periodic
GMM refits at deployment.

Pipeline:
  1. Load the production encoder (contrastive_model.pt trained on all 8 series).
  2. Extract per-series NoFlow latent centroids and covariances.
  3. Aggregate per-series mean temperatures from data/processed/<series>/*.csv
     (filter: flow_type == 'noflow' AND equilibrium == 1 AND valid == 1).
  4. Compute Pearson + Spearman correlations of centroid coordinates and
     cluster compactness against each temperature measure.
  5. 4-panel scatter plot and a summary JSON.

Run inside Docker:
    docker compose run --rm kild python scripts/analyze_drift_temperature.py

Outputs:
  - images/contrastive/drift_temp/drift_vs_temperature.png
  - data/contrastive_g5_clf/results/drift_temperature_summary.json
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contrastive.models import build_contrastive_model
from src.contrastive.train import (
    _load_splits_and_metadata,
    _prepare_patches,
    _resolve_selected_features,
    _scale_patches,
    _to_channels,
    _extract_latents,
)


SERIES_ORDER = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
    '7.03.2026', '15.03.2026', '5.04.2026', '11.04.2026',
    '18.04.2026', 'ha-25.04.2026', 'hb-26.04.2026',
]
SERIES_COLORS = [
    '#1f77b4', '#2ca02c', '#17becf', '#9467bd',
    '#ff7f0e', '#d62728', '#e377c2', '#8c564b',
    '#bcbd22', '#7f7f7f', '#000000',
]

PIPE_COL = "Temp Pipe in °C"
AMB_COL = "Temp Ambient in °C"


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get('hp', {})
    model = build_contrastive_model(
        encoder_type=hp.get('encoder_type', 'classifier'),
        input_height=ckpt.get('input_height', 9),
        input_width=ckpt.get('input_width', 32),
        in_channels=ckpt.get('n_in_channels', 2),
        channels=hp.get('cnn_channels', [32, 64, 128]),
        kernel_size=hp.get('cnn_kernel_size', 3),
        latent_dim=hp.get('latent_dim', 128),
        projection_dim=hp.get('projection_dim', 64),
        dropout=hp.get('dropout', 0.3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, hp, ckpt.get('n_in_channels', 2)


def extract_per_series_latents(data_path: Path, cfg: dict, model, device):
    """Load patches across all splits, scale with NoFlow train statistics,
    and return per-series latents + labels."""
    data_root = cfg.get('data_root', './prepared')
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(
        data_path, data_root
    )
    selected_features = _resolve_selected_features(cfg)

    all_ids = []
    for split in ('train', 'val', 'test'):
        all_ids.extend(splits_data['splits'].get(split, []))

    all_p, all_l, all_s, _ = _prepare_patches(
        all_ids, prepared_base, metadata_by_id, selected_features
    )

    train_ids = splits_data['splits']['train']
    tr_p, tr_l, _, _ = _prepare_patches(
        train_ids, prepared_base, metadata_by_id, selected_features
    )
    (_, scaled_all), _ = _scale_patches(
        tr_p, tr_l, [all_p], scaler_type=cfg.get('scaler', 'standard'),
    )

    n_feat, w = scaled_all.shape[1], scaled_all.shape[2]
    dual = cfg.get('cnn_channel_mode', 'dual') == 'dual'
    n_stats = 2 if cfg.get('include_std_features', False) else 1
    chan_all, _ = _to_channels(scaled_all, n_feat, w, n_stats, dual)

    per_series = {}
    all_s_arr = np.array(all_s)
    all_l_arr = np.array(all_l)
    for s in SERIES_ORDER:
        mask = all_s_arr == s
        if mask.sum() == 0:
            continue
        z = _extract_latents(model, chan_all[mask], device)
        per_series[s] = {'z': z, 'labels': all_l_arr[mask]}
    return per_series


def compute_latent_stats(per_series):
    """Per-series NoFlow centroid, covariance, and compactness measures."""
    stats = {}
    for s, d in per_series.items():
        nf = d['labels'] == 0
        nf_z = d['z'][nf]
        if len(nf_z) < 2:
            continue
        cov = np.cov(nf_z, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(cov.shape[0]))
        stats[s] = {
            'n_noflow': int(nf.sum()),
            'centroid': nf_z.mean(axis=0),
            'cov': cov,
            'trace_cov': float(np.trace(cov)),
            'log_det_cov': float(logdet),
            'mean_pairwise_dist': float(np.sqrt(np.trace(cov))),
        }
    return stats


def compute_per_series_temperature(data_path: Path):
    """Mean temperature per series from processed CSVs, restricted to
    NoFlow + equilibrium + valid points (same population as the NoFlow
    latents)."""
    processed_root = data_path / 'processed'
    temps = {}
    for s in SERIES_ORDER:
        series_dir = processed_root / s
        if not series_dir.exists():
            print(f"  [warn] {series_dir} missing — skipping")
            continue
        pipe_vals, amb_vals = [], []
        for csv_file in sorted(series_dir.glob('*.csv')):
            df = pd.read_csv(csv_file)
            if PIPE_COL not in df.columns or AMB_COL not in df.columns:
                continue
            mask = (df.get('flow_type', '') == 'noflow')
            if 'equilibrium' in df.columns:
                mask &= df['equilibrium'] == 1
            if 'valid' in df.columns:
                mask &= df['valid'] == 1
            if mask.sum() == 0:
                continue
            pipe_vals.extend(df.loc[mask, PIPE_COL].dropna().tolist())
            amb_vals.extend(df.loc[mask, AMB_COL].dropna().tolist())
        if not pipe_vals:
            continue
        pipe_arr, amb_arr = np.array(pipe_vals), np.array(amb_vals)
        temps[s] = {
            'n_points': len(pipe_arr),
            'temp_pipe_mean': float(pipe_arr.mean()),
            'temp_pipe_std': float(pipe_arr.std()),
            'temp_amb_mean': float(amb_arr.mean()),
            'temp_amb_std': float(amb_arr.std()),
            'temp_diff_mean': float((pipe_arr - amb_arr).mean()),
            'temp_diff_std': float((pipe_arr - amb_arr).std()),
        }
    return temps


def compute_per_series_covariates(data_path: Path):
    """Additional per-series covariates beyond mean temperature, to probe
    what drives the non-temperature-correlated latent drift component (PC1).

    Returns a dict[series] -> dict[covariate_name] = float, covering:
      - Time proxies: days_since_first, day_of_year, series_index.
      - Usage profile: noflow_fraction, flow_fraction, total_duration_days.
      - Thermal dynamics: relaxations_per_day (from relaxation_id column).

    All counts are aggregated across *all* points of the series (no
    flow-state filter), because these covariates describe the series
    context itself, not the NoFlow population.
    """
    processed_root = data_path / 'processed'
    out = {}
    first_date = None
    for idx, s in enumerate(SERIES_ORDER):
        series_dir = processed_root / s
        if not series_dir.exists():
            continue
        # Strip optional Haus-prefix (`ha-` / `hb-` / future `hc-` etc.)
        # before date parsing — the prefix encodes the installation site
        # but the trailing `dd.mm.YYYY` is still the calendar date used
        # for time-since-first / day-of-year covariates.
        s_clean = s.split('-', 1)[1] if '-' in s and s[2:3] == '-' else s
        try:
            d = datetime.datetime.strptime(s_clean, "%d.%m.%Y")
        except ValueError:
            print(f"  [warn] could not parse date from series name '{s}'")
            continue
        if first_date is None:
            first_date = d

        total_points = 0
        noflow_points = 0
        flow_points = 0
        n_relaxations = 0
        for csv_file in sorted(series_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            total_points += len(df)
            if 'flow_type' in df.columns:
                noflow_points += int((df['flow_type'] == 'noflow').sum())
                flow_points += int((df['flow_type'] == 'flow').sum())
            if 'relaxation_id' in df.columns:
                # count unique non-null relaxation IDs
                ids = df['relaxation_id'].dropna().unique()
                n_relaxations += int(len(ids))

        # Processed CSVs are ~1 Hz → points ≈ seconds
        total_days = total_points / 86400.0 if total_points > 0 else 0.0

        out[s] = {
            'days_since_first': float((d - first_date).days),
            'day_of_year': float(d.timetuple().tm_yday),
            'series_index': float(idx),
            'total_duration_days': float(total_days),
            'noflow_fraction': float(noflow_points / total_points) if total_points else 0.0,
            'flow_fraction': float(flow_points / total_points) if total_points else 0.0,
            'relaxations_per_day': float(n_relaxations / total_days) if total_days > 0 else 0.0,
            'n_relaxations': float(n_relaxations),
        }
    return out


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return float(pr), float(pp), float(sr), float(sp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=str, default='data')
    parser.add_argument('--img-path', type=str, default='images')
    parser.add_argument('--config-dir', type=str, default='configs/contrastive')
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to contrastive_model.pt (default: <output_root>/results/contrastive_model.pt)'
    )
    args = parser.parse_args()

    if os.getenv('DOCKER_ENV'):
        data_path = Path(os.getenv('CONTAINER_DATA_PATH', '/app/data'))
        img_path = Path(os.getenv('CONTAINER_IMG_PATH', '/app/images'))
        config_dir = Path(os.getenv('CONFIG_DIR_PATH', '/app/configs')) / 'contrastive'
    else:
        root = Path(__file__).resolve().parent.parent
        data_path = root / args.data_path
        img_path = root / args.img_path
        config_dir = root / args.config_dir

    out_dir = img_path / 'contrastive' / 'drift_temp'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / 'encoder.json') as f:
        cfg = json.load(f)

    output_root = cfg.get('output_root', './contrastive')
    results_dir = data_path / output_root / 'results'
    model_path = Path(args.model) if args.model else results_dir / 'contrastive_model.pt'
    if not model_path.exists():
        print(f"[error] Model not found: {model_path}")
        print("Run: python -m src.contrastive.main --run train")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading model: {model_path}")
    model, hp, n_in_ckpt = load_model(model_path, device)
    print(f"  encoder_type={hp.get('encoder_type')}, latent_dim={hp.get('latent_dim')}")

    # Variant-defining fields are not all stored in hp (only encoder_type is).
    # cnn_channel_mode and include_std_features change the input tensor shape;
    # CLI overrides on tune/train don't get written back to encoder.json, so
    # the active cfg can disagree with the trained checkpoint. Reconstruct
    # them from n_in_channels saved in the checkpoint:
    #   dual + n_in_channels=2 → mean only      (n_stats=1, std off)
    #   dual + n_in_channels=4 → mean + std     (n_stats=2, std on)
    #   single + n_in_channels=1 → mean only    (n_stats=1, std off)
    #   single + n_in_channels=2 → mean + std   (n_stats=2, std on)
    if 'encoder_type' in hp:
        cfg['encoder_type'] = hp['encoder_type']
    dual = cfg.get('cnn_channel_mode', 'dual') == 'dual'
    n_stats_inferred = n_in_ckpt // 2 if dual else n_in_ckpt
    cfg['include_std_features'] = (n_stats_inferred == 2)
    print(f"  inferred from checkpoint: dual={dual}, "
          f"include_std_features={cfg['include_std_features']} "
          f"(n_in_channels={n_in_ckpt})")

    print("\nExtracting per-series latents...")
    per_series = extract_per_series_latents(data_path, cfg, model, device)
    for s in SERIES_ORDER:
        if s in per_series:
            nf = (per_series[s]['labels'] == 0).sum()
            fl = (per_series[s]['labels'] == 1).sum()
            print(f"  {s}: {len(per_series[s]['z'])} patches (NoFlow={nf}, Flow={fl})")

    stats = compute_latent_stats(per_series)

    print("\nComputing per-series temperatures (NoFlow + equilibrium + valid)...")
    temps = compute_per_series_temperature(data_path)
    for s in SERIES_ORDER:
        if s in temps:
            t = temps[s]
            print(f"  {s}: pipe={t['temp_pipe_mean']:.2f}°C "
                  f"amb={t['temp_amb_mean']:.2f}°C "
                  f"diff={t['temp_diff_mean']:.2f}°C "
                  f"(n={t['n_points']})")

    # Series present in both
    series_both = [s for s in SERIES_ORDER if s in stats and s in temps]
    if len(series_both) < 3:
        print(f"[error] Only {len(series_both)} series have both latents and temps.")
        sys.exit(1)

    # --- PCA on centroids for 2D axes ---
    centroids = np.array([stats[s]['centroid'] for s in series_both])
    pca = PCA(n_components=2, random_state=42).fit(centroids)
    centroids_pc = pca.transform(centroids)

    temp_diff = np.array([temps[s]['temp_diff_mean'] for s in series_both])
    temp_pipe = np.array([temps[s]['temp_pipe_mean'] for s in series_both])
    temp_amb = np.array([temps[s]['temp_amb_mean'] for s in series_both])
    log_det = np.array([stats[s]['log_det_cov'] for s in series_both])
    trace_cov = np.array([stats[s]['trace_cov'] for s in series_both])

    # Mahalanobis drift from first series (10.01)
    ref_cov = stats[series_both[0]]['cov']
    lat_dim = ref_cov.shape[0]
    ref_cov_inv = np.linalg.pinv(ref_cov + 1e-6 * np.eye(lat_dim))
    ref_c = stats[series_both[0]]['centroid']
    mahal = np.array([
        float(np.sqrt((stats[s]['centroid'] - ref_c) @ ref_cov_inv @ (stats[s]['centroid'] - ref_c)))
        for s in series_both
    ])

    # Correlations: every (y, temp) pair
    targets = {
        'centroid_PC1': centroids_pc[:, 0],
        'centroid_PC2': centroids_pc[:, 1],
        'log_det_cov': log_det,
        'trace_cov': trace_cov,
        'mahalanobis_from_10.01': mahal,
    }
    temps_dict = {'temp_diff': temp_diff, 'temp_pipe': temp_pipe, 'temp_amb': temp_amb}

    corr_table = {}
    for tname, tvals in temps_dict.items():
        corr_table[tname] = {}
        for yname, yvals in targets.items():
            pr, pp, sr, sp = _corr(tvals, yvals)
            corr_table[tname][yname] = {
                'pearson_r': pr, 'pearson_p': pp,
                'spearman_r': sr, 'spearman_p': sp,
            }

    # ---- Print correlation table ----
    print("\n" + "=" * 72)
    print("CORRELATIONS (Pearson r | Spearman ρ)")
    print("=" * 72)
    print(f"{'target':<28} {'temp_diff':>14} {'temp_pipe':>14} {'temp_amb':>14}")
    for yname in targets:
        row = f"{yname:<28}"
        for tname in ('temp_diff', 'temp_pipe', 'temp_amb'):
            c = corr_table[tname][yname]
            row += f" {c['pearson_r']:+.2f}/{c['spearman_r']:+.2f}".rjust(15)
        print(row)

    # =================================================================
    # Extended covariate hunt — what drives PC1 (the temperature-free drift axis)?
    # =================================================================
    print("\nComputing additional covariates (time, usage, dynamics)...")
    extra_cov = compute_per_series_covariates(data_path)

    # Build a unified covariate matrix: per-series dict of all candidates
    # Includes the three mean-temp plus std-temp plus everything from extra_cov.
    covariate_names = [
        'temp_pipe', 'temp_amb', 'temp_diff',
        'temp_pipe_std', 'temp_amb_std', 'temp_diff_std',
        'days_since_first', 'day_of_year', 'series_index',
        'total_duration_days', 'noflow_fraction', 'flow_fraction',
        'relaxations_per_day',
    ]
    # series_both already limits to series with latents + temps; intersect with extra_cov too
    series_full = [s for s in series_both if s in extra_cov]
    if len(series_full) < 3:
        print(f"  [warn] Only {len(series_full)} series have all covariates — skipping extended matrix.")
        corr_matrix = None
    else:
        # Restrict arrays to series_full ordering
        idx_map = [series_both.index(s) for s in series_full]
        target_arrays_full = {
            'centroid_PC1': centroids_pc[idx_map, 0],
            'centroid_PC2': centroids_pc[idx_map, 1],
            'log_det_cov': log_det[idx_map],
            'trace_cov': trace_cov[idx_map],
            'mahalanobis_from_10.01': mahal[idx_map],
        }

        cov_arrays = {
            'temp_pipe': np.array([temps[s]['temp_pipe_mean'] for s in series_full]),
            'temp_amb': np.array([temps[s]['temp_amb_mean'] for s in series_full]),
            'temp_diff': np.array([temps[s]['temp_diff_mean'] for s in series_full]),
            'temp_pipe_std': np.array([temps[s]['temp_pipe_std'] for s in series_full]),
            'temp_amb_std': np.array([temps[s]['temp_amb_std'] for s in series_full]),
            'temp_diff_std': np.array([temps[s]['temp_diff_std'] for s in series_full]),
            'days_since_first': np.array([extra_cov[s]['days_since_first'] for s in series_full]),
            'day_of_year': np.array([extra_cov[s]['day_of_year'] for s in series_full]),
            'series_index': np.array([extra_cov[s]['series_index'] for s in series_full]),
            'total_duration_days': np.array([extra_cov[s]['total_duration_days'] for s in series_full]),
            'noflow_fraction': np.array([extra_cov[s]['noflow_fraction'] for s in series_full]),
            'flow_fraction': np.array([extra_cov[s]['flow_fraction'] for s in series_full]),
            'relaxations_per_day': np.array([extra_cov[s]['relaxations_per_day'] for s in series_full]),
        }

        # Compute full Spearman matrix (more robust than Pearson for n=8)
        corr_matrix = np.full((len(covariate_names), len(target_arrays_full)), np.nan)
        corr_pearson = np.full_like(corr_matrix, np.nan)
        target_list = list(target_arrays_full.keys())
        for i, cname in enumerate(covariate_names):
            for j, tname in enumerate(target_list):
                pr, _, sr, _ = _corr(cov_arrays[cname], target_arrays_full[tname])
                corr_matrix[i, j] = sr
                corr_pearson[i, j] = pr

        # Print the full table
        print("\n" + "=" * 100)
        print("EXTENDED COVARIATE MATRIX — Spearman ρ (Pearson r in parens)")
        print(f"  n_series = {len(series_full)}")
        print("=" * 100)
        header = f"{'covariate':<24}"
        for t in target_list:
            header += f" {t:>20}"
        print(header)
        for i, cname in enumerate(covariate_names):
            row = f"{cname:<24}"
            for j, tname in enumerate(target_list):
                row += f"  {corr_matrix[i, j]:+.2f}({corr_pearson[i, j]:+.2f})".rjust(21)
            print(row)

        # Best predictor for each target
        print("\n" + "-" * 72)
        print("BEST COVARIATE PER TARGET (by |Spearman ρ|)")
        print("-" * 72)
        for j, tname in enumerate(target_list):
            col = np.abs(corr_matrix[:, j])
            best_i = int(np.nanargmax(col))
            rho = corr_matrix[best_i, j]
            r = corr_pearson[best_i, j]
            verdict = (
                "strong link — single-covariate model viable" if abs(rho) > 0.8 else
                "moderate — multi-covariate needed" if abs(rho) > 0.5 else
                "no strong link — this target needs more data or unknown covariate"
            )
            print(f"  {tname:<26} → {covariate_names[best_i]:<22} "
                  f"ρ={rho:+.2f}, r={r:+.2f}   [{verdict}]")

        # =================================================================
        # Correlation heatmap
        # =================================================================
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax3.set_xticks(np.arange(len(target_list)))
        ax3.set_xticklabels(target_list, rotation=45, ha='right', fontsize=10)
        ax3.set_yticks(np.arange(len(covariate_names)))
        ax3.set_yticklabels(covariate_names, fontsize=10)
        # Annotate cells with ρ
        for i in range(len(covariate_names)):
            for j in range(len(target_list)):
                val = corr_matrix[i, j]
                if np.isfinite(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax3.text(j, i, f"{val:+.2f}", ha='center', va='center',
                             color=color, fontsize=8, fontweight='bold')
        # Separator line between temperature block and other covariates
        ax3.axhline(5.5, color='black', lw=1.5)
        ax3.axhline(8.5, color='black', lw=0.8, ls='--')
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Spearman ρ', fontsize=11)
        ax3.set_title(
            f'Covariate ↔ Drift Correlation (n={len(series_full)} series)\n'
            'Top block: temperature — Middle: time — Bottom: usage & dynamics',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()
        out_png3 = out_dir / 'covariate_correlation_matrix.png'
        fig3.savefig(out_png3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"\nSaved: {out_png3}")

        # PC1-focused scatter panel (top-4 PC1 covariates)
        pc1_col = np.abs(corr_matrix[:, target_list.index('centroid_PC1')])
        top4_pc1 = np.argsort(-pc1_col)[:4]
        fig4, axes4 = plt.subplots(2, 2, figsize=(13, 10))
        pc1_vals = target_arrays_full['centroid_PC1']
        colors_full = [SERIES_COLORS[SERIES_ORDER.index(s)] for s in series_full]
        for k, cov_i in enumerate(top4_pc1):
            ax = axes4[k // 2, k % 2]
            xvals = cov_arrays[covariate_names[cov_i]]
            for ii, s in enumerate(series_full):
                ax.scatter(xvals[ii], pc1_vals[ii], s=130, color=colors_full[ii],
                           edgecolors='black', linewidth=1.2, zorder=3)
                ax.annotate(s[:5], (xvals[ii], pc1_vals[ii]), fontsize=8,
                            xytext=(5, 5), textcoords='offset points')
            if len(xvals) >= 3:
                slope, intercept = np.polyfit(xvals, pc1_vals, 1)
                xline = np.linspace(xvals.min(), xvals.max(), 50)
                ax.plot(xline, slope * xline + intercept, 'k--', alpha=0.4, lw=1)
            pr, _, sr, _ = _corr(xvals, pc1_vals)
            ax.set_xlabel(covariate_names[cov_i])
            ax.set_ylabel('Centroid PC1')
            ax.set_title(f'{covariate_names[cov_i]}  r={pr:+.2f}, ρ={sr:+.2f}', fontsize=10)
            ax.grid(True, alpha=0.3)
        fig4.suptitle('Top-4 covariates ranked against PC1 (the temperature-free drift axis)',
                      fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_png4 = out_dir / 'pc1_best_covariates.png'
        fig4.savefig(out_png4, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"Saved: {out_png4}")

    # =================================================================
    # Main 4-panel plot — decision basis for drift modelling
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = [SERIES_COLORS[SERIES_ORDER.index(s)] for s in series_both]

    def _panel(ax, x, y, xlabel, ylabel, title_prefix):
        for i, s in enumerate(series_both):
            ax.scatter(x[i], y[i], s=140, color=colors[i],
                       edgecolors='black', linewidth=1.2, zorder=3)
            ax.annotate(s[:5], (x[i], y[i]), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')
        # Linear fit line
        if len(x) >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 50)
            ax.plot(xline, slope * xline + intercept, 'k--', alpha=0.4, lw=1)
        pr, _, sr, _ = _corr(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}\nPearson r={pr:+.2f}, Spearman ρ={sr:+.2f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3)

    _panel(axes[0, 0], temp_diff, centroids_pc[:, 0],
           'temp_diff mean [°C]', f'Centroid PC1 ({pca.explained_variance_ratio_[0]:.0%})',
           'NoFlow centroid PC1 vs Temp Diff')
    _panel(axes[0, 1], temp_diff, centroids_pc[:, 1],
           'temp_diff mean [°C]', f'Centroid PC2 ({pca.explained_variance_ratio_[1]:.0%})',
           'NoFlow centroid PC2 vs Temp Diff')
    _panel(axes[1, 0], temp_diff, log_det,
           'temp_diff mean [°C]', 'log det Σ (compactness)',
           'NoFlow cluster size vs Temp Diff')
    _panel(axes[1, 1], temp_diff, mahal,
           'temp_diff mean [°C]', f'Mahalanobis dist from {series_both[0]}',
           'Drift magnitude vs Temp Diff')

    fig.suptitle('Drift vs. Temperature — NoFlow cluster across 8 series',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_png = out_dir / 'drift_vs_temperature.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_png}")

    # =================================================================
    # Secondary plot: centroid trajectory in PCA, node color=temp_diff,
    #                  node size=trace_cov  (all-in-one diagnostic)
    # =================================================================
    fig2, ax = plt.subplots(figsize=(10, 8))
    sizes = 100 + 900 * (trace_cov - trace_cov.min()) / max(float(np.ptp(trace_cov)), 1e-9)
    sc = ax.scatter(centroids_pc[:, 0], centroids_pc[:, 1],
                    c=temp_diff, cmap='coolwarm', s=sizes,
                    edgecolors='black', linewidth=1.2, zorder=3)
    for i in range(len(series_both) - 1):
        ax.annotate('', xy=centroids_pc[i + 1], xytext=centroids_pc[i],
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.6))
    for i, s in enumerate(series_both):
        ax.annotate(s[:5], centroids_pc[i], fontsize=9,
                    xytext=(8, 8), textcoords='offset points')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('temp_diff (°C)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
    ax.set_title('Centroid trajectory (color=temp, size=trace Σ, arrows=time)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png2 = out_dir / 'centroid_trajectory_temp.png'
    fig2.savefig(out_png2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out_png2}")

    # =================================================================
    # Summary JSON
    # =================================================================
    summary = {
        'model_path': str(model_path),
        'encoder_type': hp.get('encoder_type'),
        'latent_dim': hp.get('latent_dim'),
        'series': {},
        'correlations': corr_table,
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    }
    for i, s in enumerate(series_both):
        summary['series'][s] = {
            'n_noflow': stats[s]['n_noflow'],
            'centroid_pc1': float(centroids_pc[i, 0]),
            'centroid_pc2': float(centroids_pc[i, 1]),
            'trace_cov': stats[s]['trace_cov'],
            'log_det_cov': stats[s]['log_det_cov'],
            'mahalanobis_from_ref': float(mahal[i]),
            'temp_pipe_mean': temps[s]['temp_pipe_mean'],
            'temp_amb_mean': temps[s]['temp_amb_mean'],
            'temp_diff_mean': temps[s]['temp_diff_mean'],
            'temp_diff_std': temps[s]['temp_diff_std'],
        }

    # Attach extended covariate values + full Spearman/Pearson matrix
    if corr_matrix is not None:
        summary['extended_covariates'] = {
            s: {
                **{k: float(v) for k, v in extra_cov[s].items()},
                'temp_pipe_std': temps[s]['temp_pipe_std'],
                'temp_amb_std': temps[s]['temp_amb_std'],
                'temp_diff_std': temps[s]['temp_diff_std'],
            }
            for s in series_full
        }
        summary['covariate_correlation_matrix'] = {
            'covariates': covariate_names,
            'targets': target_list,
            'spearman': corr_matrix.tolist(),
            'pearson': corr_pearson.tolist(),
            'n_series': len(series_full),
        }

    summary_path = results_dir / 'drift_temperature_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # ---- Decision hint ----
    print("\n" + "=" * 72)
    print("INTERPRETATION HINT")
    print("=" * 72)
    print("  |ρ| > 0.8 : single covariate explains this target — viable model input.")
    print("  0.5 < |ρ| ≤ 0.8 : partial — combine with another covariate.")
    print("  |ρ| ≤ 0.5 : no useful link in the current covariate set.")
    print("  For PC1 specifically: if no covariate reaches 0.8, more data or ")
    print("  additional sensor signals are needed to close the drift model.")
    print("=" * 72)


if __name__ == '__main__':
    main()
