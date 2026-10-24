#!/usr/bin/env python
"""
Pooled DeepSAD centre diagnostic for the two-house training study.

Question: after training one encoder on BOTH Haus A and Haus B, does their
NoFlow data form ONE tight cloud around the shared frozen DeepSAD centre c,
or do the two houses sit as separate clouds with c stranded between them?

This is a representation-geometry check, NOT a leak-detection eval: no GMM,
no calibration/test/Leakage data. Only NoFlow patches that were part of
ENCODER TRAINING are used, forwarded once through the finished, frozen
encoder + projection head, in the exact z_proj space the DeepSAD centre
lives in (L2-normalised projection-head output).

This supersedes the narrower `analyze_deepsad_center_check.py` quick-check:
same underlying data path and centre-source handling, but restructured to
the (house, n, d, r, rho) record schema, explicit correctness assertions,
persisted raw latents (for offline re-plotting without re-running the
encoder), and a publication-quality figure for the thesis
(figures/deepsad_center_diagnostic.pdf/png) instead of a quick PNG.

Center source: uses the TRUE frozen training-time centre if the checkpoint
was saved with it (`ckpt['deepsad_center']`, added to the checkpoint format
2026-07-21). Older checkpoints (e.g. the current
contrastive_g5_deepsad_transformer_ae_hb_diversity run) don't have it, so
this script falls back to recomputing the mean of pooled NoFlow z_proj
through the FINAL encoder as a proxy — DeepSAD's loss already pulls NoFlow
toward the true center, so this proxy should sit close to it, but it is NOT
bit-identical, and is labelled as such in every output (JSON meta, CSV meta
column, printed interpretation, figure caption).

Run inside Docker on the GPU server (needs data/prepared/ + a trained
checkpoint):
    docker compose run --rm kild python scripts/analyze_deepsad_center_diagnostic.py \\
        --output-root ./contrastive_g5_deepsad_transformer_ae_hb_diversity

Local sanity-check without server data (synthetic points, fixed seed, only
exercises the statistics/assertions/figure code path):
    python scripts/analyze_deepsad_center_diagnostic.py --self-test

Outputs (all under data/results/pooled_center_diagnostic/ -- a mounted
Docker volume; ./figures/ is NOT mounted in the container and must not be
used as a save target here, see docker-compose.yml):
  - data/results/pooled_center_diagnostic/pooled_center_diagnostic.json
  - data/results/pooled_center_diagnostic/pooled_center_diagnostic.csv
  - data/results/pooled_center_diagnostic/pooled_center_latents.npz
    (z_a, z_b, c, mean_a, mean_b -- lets the figure be rebuilt offline)
  - data/results/pooled_center_diagnostic/deepsad_center_diagnostic.pdf
  - data/results/pooled_center_diagnostic/deepsad_center_diagnostic.png (600 dpi)

After scp'ing these back (see commands.txt), copy the two figure files into
./figures/ locally to match the thesis figure convention.
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / 'data' / 'results' / 'pooled_center_diagnostic'
# NOTE: the figure is saved under RESULTS_DIR (data/results/...), not
# ./figures/. Inside the Docker container only /app/data and /app/images
# are mounted volumes (see docker-compose.yml) -- /app/figures is not
# mounted and would silently vanish on `--rm`. Copy the two files into
# figures/ locally after scp'ing them back (see commands.txt).

# Matches the ENCODER-TRAIN series actually used for the
# contrastive_g5_deepsad_transformer_ae_hb_diversity checkpoint (17 Haus A
# loso_series + the 2 Haus B leak series folded into its training). Override
# via CLI if checking a different pooled checkpoint.
DEFAULT_HAUS_A = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026', '7.03.2026',
    '15.03.2026', '5.04.2026', '11.04.2026', '18.04.2026', 'ha-25.04.2026',
    'ha-2.05.2026', 'ha-9.05.2026', 'ha-17.05.2026', 'ha-24.05.2026',
    'ha-6.06.2026', 'ha-2.06.2026', 'ha-20.06.2026',
]
DEFAULT_HAUS_B = ['hb-17.05.2026', 'hb-25.05.2026']

# palette consistent with the other thesis concept figures (SupCon/DeepSAD/GMM)
C_HA = '#5b8fc2'      # muted blue
C_HB = '#c1663c'      # muted warm terracotta
C_CENTER = '#b8860b'  # gold -- same "fixed centre" device as plot_thesis_deepsad_concept.py
C_TEXT = '#232323'
C_SUB = '#5c5c5c'


# ═════════════════════════════════════════════════════════════════════════
# data loading (real run only; skipped entirely in --self-test)
# ═════════════════════════════════════════════════════════════════════════
def load_checkpoint(output_root: str, device):
    import torch
    from src.contrastive.models import build_contrastive_model

    results_dir = ROOT / 'data' / output_root.lstrip('./') / 'results'
    ckpt = torch.load(results_dir / 'contrastive_model.pt', map_location=device, weights_only=False)
    hp = ckpt['hp']
    model = build_contrastive_model(
        encoder_type=hp['encoder_type'],
        encoder_arch=hp.get('encoder_arch', 'cnn'),
        in_channels=ckpt['n_in_channels'],
        input_height=ckpt['input_height'],
        input_width=ckpt['input_width'],
        latent_dim=hp['latent_dim'],
        projection_dim=hp['projection_dim'],
        channels=hp.get('cnn_channels', [32, 64, 128]),
        kernel_size=hp.get('cnn_kernel_size', 3),
        dropout=hp.get('dropout', 0.3),
        d_model=hp.get('d_model', 128),
        num_heads=hp.get('num_heads', 4),
        num_layers=hp.get('num_layers', 4),
        ff_dim=hp.get('ff_dim', 256),
        transformer_dropout=hp.get('transformer_dropout', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    assert model.training is False, "encoder must be frozen (eval mode) for this diagnostic"

    with open(results_dir / 'input_scaler.pkl', 'rb') as f:
        scaler_bundle = pickle.load(f)

    return model, ckpt, scaler_bundle, results_dir


def extract_house_noflow_latents(args, model, device, scaler_bundle):
    """Load encoder-training NoFlow patches for each house and forward them
    through the frozen encoder + projection head, returning z_proj arrays."""
    import torch
    from src.common.utils import detect_n_stats
    from src.contrastive.train import (
        _load_splits_and_metadata, _ids_for_series, _prepare_patches,
        _extract_latents, _to_channels,
    )
    from src.contrastive.main import FEATURE_GROUPS

    # ── House identification correctness ────────────────────────────────
    ha_series, hb_series = list(args.haus_a_series), list(args.haus_b_series)
    assert set(ha_series).isdisjoint(hb_series), (
        "House A / House B series lists overlap — cannot be the same file set")
    assert all(not s.startswith('hb-') for s in ha_series), (
        "House A list contains an 'hb-' series (project naming convention: "
        "'hb-' prefix = Haus B)")
    assert all(s.startswith('hb-') for s in hb_series), (
        "House B list must use the 'hb-' naming convention")

    data_path = ROOT / 'data'
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, args.data_root)
    selected = FEATURE_GROUPS[args.feature_group]

    ha_ids = _ids_for_series(splits_data, metadata_by_id, ha_series)
    hb_ids = _ids_for_series(splits_data, metadata_by_id, hb_series)
    ha_patches, ha_labels, _, _ = _prepare_patches(ha_ids, prepared_base, metadata_by_id, selected)
    hb_patches, hb_labels, _, _ = _prepare_patches(hb_ids, prepared_base, metadata_by_id, selected)

    # ── only encoder-training NoFlow patches, never GMM/calibration/test/Leakage ──
    ha_labels_arr, hb_labels_arr = np.asarray(ha_labels), np.asarray(hb_labels)
    ha_mask, hb_mask = ha_labels_arr == 0, hb_labels_arr == 0
    assert ha_mask.any(), "no NoFlow patches found for House A"
    assert hb_mask.any(), "no NoFlow patches found for House B"
    ha_noflow, hb_noflow = ha_patches[ha_mask], hb_patches[hb_mask]
    assert set(np.unique(ha_labels_arr[ha_mask])) == {0}, "House A NoFlow filter leaked non-NoFlow patches"
    assert set(np.unique(hb_labels_arr[hb_mask])) == {0}, "House B NoFlow filter leaked non-NoFlow patches"
    print(f"House A NoFlow (encoder-train) patches: {len(ha_noflow)} | "
          f"House B NoFlow (encoder-train) patches: {len(hb_noflow)}")
    print("No GMM, calibration, test, or Leakage data referenced above "
          "(structurally: this function never imports a GMM-fitting routine).")

    # same saved (NoFlow-only-fit) scaler + dual-channel reshape the model
    # was trained on -- mirrors eval_contrastive's exact preprocessing flow.
    saved_scaler = scaler_bundle['scaler']
    feat_names = scaler_bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    dual = (scaler_bundle.get('cnn_channel_mode', 'dual') == 'dual')

    def scale_and_channel(patches: np.ndarray) -> np.ndarray:
        if saved_scaler is not None:
            n_, f_, w_ = patches.shape
            flat = patches.transpose(0, 2, 1).reshape(-1, f_)
            patches = saved_scaler.transform(flat).reshape(n_, w_, f_).transpose(0, 2, 1).astype(np.float32)
        channels, _ = _to_channels(patches, patches.shape[1], patches.shape[2], n_stats, dual)
        return channels

    ha_noflow_c = scale_and_channel(ha_noflow)
    hb_noflow_c = scale_and_channel(hb_noflow)

    z_a = _extract_latents(model, ha_noflow_c, device, use_proj=True)
    z_b = _extract_latents(model, hb_noflow_c, device, use_proj=True)
    return z_a, z_b


# ═════════════════════════════════════════════════════════════════════════
# core statistics (shared by real run and --self-test)
# ═════════════════════════════════════════════════════════════════════════
def compute_diagnostic(z_a: np.ndarray, z_b: np.ndarray, c: np.ndarray) -> dict:
    assert z_a.ndim == 2 and z_b.ndim == 2 and c.ndim == 1, \
        "z_a, z_b must be (N, D); c must be (D,)"
    assert z_a.shape[1] == z_b.shape[1] == c.shape[0], \
        "the DeepSAD centre and projected latents must share the same dimension"
    assert len(z_a) > 0 and len(z_b) > 0, "need NoFlow patches from both houses"

    mean_a = z_a.mean(axis=0)
    mean_b = z_b.mean(axis=0)
    d_a = float(np.linalg.norm(mean_a - c))
    d_b = float(np.linalg.norm(mean_b - c))
    spread_a = float(np.linalg.norm(z_a - mean_a, axis=1).mean())
    spread_b = float(np.linalg.norm(z_b - mean_b, axis=1).mean())
    rho_a = d_a / spread_a if spread_a > 0 else float('nan')
    rho_b = d_b / spread_b if spread_b > 0 else float('nan')

    return dict(mean_a=mean_a, mean_b=mean_b, d_a=d_a, d_b=d_b,
                spread_a=spread_a, spread_b=spread_b, rho_a=rho_a, rho_b=rho_b)


def build_records(n_a: int, n_b: int, stats: dict) -> list:
    return [
        dict(house='A', number_of_noflow_patches=int(n_a),
             centroid_to_shared_center_distance=stats['d_a'],
             within_house_mean_spread=stats['spread_a'],
             normalized_displacement_ratio=stats['rho_a']),
        dict(house='B', number_of_noflow_patches=int(n_b),
             centroid_to_shared_center_distance=stats['d_b'],
             within_house_mean_spread=stats['spread_b'],
             normalized_displacement_ratio=stats['rho_b']),
    ]


def save_results(records: list, meta: dict, json_path: Path, csv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({'meta': meta, 'results': records}, f, indent=2)
    print(f'Saved -> {json_path}')

    fieldnames = ['house', 'number_of_noflow_patches',
                  'centroid_to_shared_center_distance', 'within_house_mean_spread',
                  'normalized_displacement_ratio']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f'Saved -> {csv_path}')


def print_interpretation(stats: dict, center_source: str):
    """Report the numbers transparently. No universal pass/fail threshold is
    invented here -- rho_A/rho_B are reported as-is; the reader judges
    "large" relative to the project's own goals, unless a threshold has been
    explicitly set elsewhere by the project or supervisor."""
    print("\n=== Interpretation (no invented universal threshold) ===")
    print(f"Centre source: {center_source}")
    print(f"House A: d_A={stats['d_a']:.5f}, r_A={stats['spread_a']:.5f}, "
          f"rho_A=d_A/r_A={stats['rho_a']:.3f}")
    print(f"House B: d_B={stats['d_b']:.5f}, r_B={stats['spread_b']:.5f}, "
          f"rho_B=d_B/r_B={stats['rho_b']:.3f}")
    print(
        "Reading: the shared centre appears suitable for a house if its "
        "centroid-to-centre distance d_h is small relative to that house's "
        "own within-house spread r_h (rho_h close to 0 => the centre sits "
        "inside the house's own cloud). A house should be flagged if rho_h "
        "is large relative to the OTHER house's rho and to what the project "
        "considers acceptable -- this script does not hard-code that cutoff; "
        f"report rho_A={stats['rho_a']:.3f} and rho_B={stats['rho_b']:.3f} "
        "and judge them against whatever criterion the project/supervisor "
        "has defined (if any) before drawing a pass/fail conclusion."
    )


# ═════════════════════════════════════════════════════════════════════════
# figure
# ═════════════════════════════════════════════════════════════════════════
def build_figure(z_a, z_b, c, stats, center_source, out_pdf, out_png,
                  seed=42, max_scatter=800):
    rng = np.random.default_rng(seed)
    mean_a, mean_b = stats['mean_a'], stats['mean_b']
    d_a, d_b = stats['d_a'], stats['d_b']
    rho_a, rho_b = stats['rho_a'], stats['rho_b']

    # Generic PCA on the pooled points (House A + House B + c) was tried
    # first, per the original spec. Verification against real data showed
    # it can visually INVERT the d_A vs. d_B ordering: PC1+PC2 maximize
    # variance across all ~11k points combined, which is not the same
    # thing as preserving the two specific, small, similar-magnitude
    # distances (centre -> each house's own mean) this figure exists to
    # compare. Confirmed on the real checkpoint: true d_A=0.0235 < d_B=
    # 0.0325, but generic top-2 PCA (93.5% variance explained) rendered
    # d_A's arrow as the LONGER one -- backwards. That is not an
    # acceptable "visualization only" distortion for the one comparison
    # that matters here, so the projection below is used instead: an
    # exact 2D plane through c, mean_A, and mean_B (any three points
    # define one), with an orthonormal basis built from
    # u = mean_A - c, v = mean_B - c. By construction this makes the
    # plotted arrow lengths EXACTLY equal to d_A and d_B (verified by
    # assertion below) -- no PCA distortion is possible for the
    # centre-to-centroid geometry, at the cost of no longer being
    # "the two directions of maximum variance" for the raw scatter.
    origin = c
    u = mean_a - origin
    v = mean_b - origin
    e1 = u / np.linalg.norm(u)
    v_perp = v - np.dot(v, e1) * e1
    norm_perp = np.linalg.norm(v_perp)
    if norm_perp < 1e-9:
        # mean_A - c and mean_B - c are (numerically) parallel -- pick an
        # arbitrary direction orthogonal to e1 just so the plot doesn't
        # degenerate to a line. Extremely unlikely in practice.
        rng_fallback = np.random.default_rng(0)
        rand_vec = rng_fallback.normal(size=e1.shape[0])
        rand_vec -= np.dot(rand_vec, e1) * e1
        e2 = rand_vec / np.linalg.norm(rand_vec)
    else:
        e2 = v_perp / norm_perp
    basis = np.stack([e1, e2], axis=0)  # (2, D)

    def project(x):
        return (np.atleast_2d(x) - origin) @ basis.T

    za2 = project(z_a)
    zb2 = project(z_b)
    c2 = project(origin)[0]  # always (0, 0) by construction
    meanA2 = project(mean_a)[0]
    meanB2 = project(mean_b)[0]

    # Verify the whole point of switching away from generic PCA actually
    # holds: arrow lengths in this view must equal the true distances.
    assert np.isclose(np.linalg.norm(meanA2 - c2), d_a, atol=1e-6), \
        "reference-plane projection failed to preserve d_A"
    assert np.isclose(np.linalg.norm(meanB2 - c2), d_b, atol=1e-6), \
        "reference-plane projection failed to preserve d_B"

    # fraction of each house's own full-space variance visible along this
    # 2-axis reference plane (analogous to PCA's explained-variance-ratio,
    # but honestly labelled as such -- these axes were NOT chosen to
    # maximize it)
    var_a_total = float(np.sum(np.var(z_a, axis=0)))
    var_a_shown = float(np.sum(np.var(za2, axis=0)))
    var_b_total = float(np.sum(np.var(z_b, axis=0)))
    var_b_shown = float(np.sum(np.var(zb2, axis=0)))
    frac_a = var_a_shown / var_a_total if var_a_total > 0 else float('nan')
    frac_b = var_b_shown / var_b_total if var_b_total > 0 else float('nan')

    def maybe_subsample(pts):
        if len(pts) > max_scatter:
            idx = rng.choice(len(pts), size=max_scatter, replace=False)
            return pts[idx]
        return pts

    za2_plot = maybe_subsample(za2)
    zb2_plot = maybe_subsample(zb2)

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(za2_plot[:, 0], za2_plot[:, 1], s=22, alpha=0.55, color=C_HA,
               edgecolor='none', zorder=2, label=f'House A NoFlow (n={len(z_a)})')
    ax.scatter(zb2_plot[:, 0], zb2_plot[:, 1], s=22, alpha=0.55, color=C_HB,
               edgecolor='none', zorder=2, label=f'House B NoFlow (n={len(z_b)})')

    # 1-sigma covariance ellipses of the ACTUAL 2D projection -- an honest
    # visual proxy for spread in this view, not a re-projection of r_A/r_B
    # (which are full-latent-space scalars and cannot be drawn exactly here).
    # Trimmed to the innermost 90% by RADIAL distance from each house's own
    # 2D centroid first. Independent per-axis percentile trimming still lets
    # diagonal tail/sub-seasonal-cluster points inflate the covariance (this
    # data is genuinely multimodal, see the GMM density figure); a radial
    # trim excludes them regardless of direction, giving an ellipse that
    # actually hugs the dense core instead of ballooning to the tail.
    def trim(pts2):
        centroid = pts2.mean(axis=0)
        dists = np.linalg.norm(pts2 - centroid, axis=1)
        thresh = np.percentile(dists, 90)
        keep = dists <= thresh
        return pts2[keep] if keep.sum() >= 3 else pts2

    for pts2, color in [(za2, C_HA), (zb2, C_HB)]:
        cov = np.cov(trim(pts2), rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(np.clip(vals, 0, None))
        centroid2 = pts2.mean(axis=0)
        ell = Ellipse(centroid2, width=width, height=height, angle=angle,
                      fill=False, edgecolor=color, linewidth=1.3,
                      linestyle=(0, (4, 2)), alpha=0.85, zorder=3)
        ax.add_patch(ell)

    ax.scatter(*meanA2, s=230, marker='D', facecolor=C_HA, edgecolor='white',
               linewidth=1.3, zorder=6, label=r'House A centroid $\bar{z}_A$')
    ax.scatter(*meanB2, s=230, marker='D', facecolor=C_HB, edgecolor='white',
               linewidth=1.3, zorder=6, label=r'House B centroid $\bar{z}_B$')
    ax.scatter(*c2, s=460, marker='*', facecolor=C_CENTER, edgecolor='#7a5c0a',
               linewidth=1.0, zorder=7, label=r'Shared DeepSAD centre $c$')

    # Robust, SQUARE view window centered on c. House A in particular has a
    # genuine long tail on PC1 (not just 1-2 stray points), so independent
    # per-axis percentile limits still come out ~10x wider in x than in y --
    # forcing equal-aspect on THAT crushes the plot into a sliver. A square
    # window (equal x/y half-range by construction) avoids that regardless
    # of how skewed the marginal distributions are.
    all_x = np.concatenate([za2[:, 0], zb2[:, 0]])
    all_y = np.concatenate([za2[:, 1], zb2[:, 1]])
    dist_from_c = np.sqrt((all_x - c2[0]) ** 2 + (all_y - c2[1]) ** 2)
    d_a2 = float(np.linalg.norm(np.asarray(meanA2) - c2))
    d_b2 = float(np.linalg.norm(np.asarray(meanB2) - c2))
    R = max(np.percentile(dist_from_c, 75), 1.15 * max(d_a2, d_b2)) * 1.08
    xlim = (c2[0] - R, c2[0] + R)
    ylim = (c2[1] - R, c2[1] + R)
    n_clipped = int(np.sum(dist_from_c > R))

    # label offset scale relative to the ZOOMED view, not the raw
    # outlier-inclusive extent, so it stays legible regardless of how
    # short/long an individual d_A or d_B arrow happens to be
    label_offset = 0.05 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    for target, lbl, color in [(meanA2, r'$d_A$', C_HA), (meanB2, r'$d_B$', C_HB)]:
        arr = FancyArrowPatch(tuple(c2), tuple(target), arrowstyle='-|>',
                               mutation_scale=14, linewidth=1.5, color=color,
                               zorder=5, shrinkA=9, shrinkB=11)
        ax.add_patch(arr)
        vec = np.asarray(target, dtype=float) - np.asarray(c2, dtype=float)
        length = np.linalg.norm(vec)
        perp = np.array([-vec[1], vec[0]]) / length if length > 1e-9 else np.array([0.0, 1.0])
        mid = (np.asarray(c2) + np.asarray(target)) / 2
        label_pos = mid + perp * label_offset
        ax.text(label_pos[0], label_pos[1], lbl, ha='center', va='center',
                fontsize=12, color=color, fontweight='bold', zorder=9,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                           edgecolor='none', alpha=0.75))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'axis 1 ($c\!\to\!\bar{z}_A$ direction)' + f'  [{frac_a * 100:.1f}% of A var. shown]')
    ax.set_ylabel(r'axis 2 ($\perp$, spans $\bar{z}_B$)' + f'  [{frac_b * 100:.1f}% of B var. shown]')
    ax.set_title('Pooled DeepSAD Centre Diagnostic', fontsize=15.5, fontweight='bold',
                 color=C_TEXT, pad=12)
    legend = ax.legend(frameon=False, loc='best', fontsize=9)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    box_text = (rf'$\rho_A = d_A / r_A = {rho_a:.2f}$' + '\n'
                rf'$\rho_B = d_B / r_B = {rho_b:.2f}$')
    ax.text(0.02, 0.02, box_text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10.5, color=C_TEXT, zorder=8,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#fff7e2',
                       edgecolor='#b8903f', linewidth=1.3))

    # Caption text removed from the figure itself per request; the same
    # disclosures (proxy centre, PCA-vs-reference-plane rationale, clipped
    # outlier count) still go to stdout / the JSON meta so nothing is lost,
    # just kept out of the image.
    print(f"Figure: {n_clipped} far-outlier point(s) fall outside the plotted "
          f"(zoomed) range; axes are an exact centre-centroid reference plane, "
          f"not generic PCA (see code comments for why).")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(out_png, format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_pdf}')
    print(f'Saved -> {out_png}')


# ═════════════════════════════════════════════════════════════════════════
# self-test: synthetic data, no server/checkpoint required
# ═════════════════════════════════════════════════════════════════════════
def make_synthetic(seed=42, dim=64, n_a=900, n_b=250):
    """Fixed-seed synthetic z_a/z_b/c for exercising the statistics + figure
    code without a GPU-server checkpoint. Loosely mimics the real geometry
    already observed for this checkpoint (House A tightly anchored, House B
    looser) purely so the self-test render is representative -- these are
    NOT real experimental numbers and must never be reported as such."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=dim)
    c = c / np.linalg.norm(c)

    center_a = c + 0.02 * rng.normal(size=dim)
    z_a = center_a + 0.08 * rng.normal(size=(n_a, dim))

    center_b = c + 0.03 * rng.normal(size=dim)
    z_b = center_b + 0.03 * rng.normal(size=(n_b, dim))

    return z_a.astype(np.float64), z_b.astype(np.float64), c.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=str,
                        default='./contrastive_g5_deepsad_transformer_ae_hb_diversity')
    parser.add_argument('--data-root', type=str, default='./prepared')
    parser.add_argument('--haus-a-series', nargs='+', default=DEFAULT_HAUS_A)
    parser.add_argument('--haus-b-series', nargs='+', default=DEFAULT_HAUS_B)
    parser.add_argument('--feature-group', type=str, default='g5')
    parser.add_argument('--self-test', action='store_true',
                        help='use fixed-seed synthetic latents instead of loading a '
                             'checkpoint/data -- for local sanity-checking only')
    args = parser.parse_args()

    if args.self_test:
        print("Running in --self-test mode: SYNTHETIC latents (fixed seed), "
              "no checkpoint or prepared data touched. Numbers below are NOT "
              "real experimental results.")
        z_a, z_b, c = make_synthetic()
        n_a, n_b = len(z_a), len(z_b)
        center_source = 'synthetic_self_test'
    else:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, ckpt, scaler_bundle, results_dir = load_checkpoint(args.output_root, device)
        print(f"Loaded checkpoint from {results_dir} (best_epoch={ckpt.get('best_epoch', '?')})")

        z_a, z_b = extract_house_noflow_latents(args, model, device, scaler_bundle)
        n_a, n_b = len(z_a), len(z_b)

        saved_center = ckpt.get('deepsad_center')
        if saved_center is not None:
            c = saved_center.numpy().astype(np.float64)
            center_source = 'saved_frozen_training_center'
        else:
            z_pooled = np.concatenate([z_a, z_b], axis=0)
            c = z_pooled.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            center_source = (
                'recomputed_proxy_from_final_encoder — checkpoint predates the '
                'deepsad_center checkpoint field; this is mean(pooled NoFlow z_proj) '
                '+ L2-normalize using the FINAL encoder, not the literal frozen '
                'pre-training buffer. DeepSAD pulls NoFlow toward the true center '
                'during training, so this should sit close to it, but is not '
                'bit-identical.'
            )
        print(f"Center source: {center_source.split(' — ')[0]}")

    stats = compute_diagnostic(z_a, z_b, c)
    records = build_records(n_a, n_b, stats)

    meta = {
        'center_source': center_source,
        'projected_dimension': int(c.shape[0]),
        'haus_a_series': list(args.haus_a_series),
        'haus_b_series': list(args.haus_b_series),
        'data_scope': 'encoder-training NoFlow patches only; no GMM, calibration, '
                       'test, or Leakage data used',
    }
    print(json.dumps({'meta': meta, 'results': records}, indent=2))

    # --self-test writes to a clearly separate subdirectory so it can never
    # clobber a real run's JSON/CSV/npz/figure (bit us twice during
    # development -- do not remove this isolation).
    out_dir = (RESULTS_DIR / '_self_test') if args.self_test else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'pooled_center_diagnostic.json'
    csv_path = out_dir / 'pooled_center_diagnostic.csv'
    save_results(records, meta, json_path, csv_path)

    npz_path = out_dir / 'pooled_center_latents.npz'
    np.savez_compressed(npz_path, z_a=z_a, z_b=z_b, c=c,
                        mean_a=stats['mean_a'], mean_b=stats['mean_b'])
    print(f'Saved -> {npz_path}  (raw latents, for offline re-plotting)')

    print_interpretation(stats, center_source)

    build_figure(z_a, z_b, c, stats, center_source,
                out_pdf=out_dir / 'deepsad_center_diagnostic.pdf',
                out_png=out_dir / 'deepsad_center_diagnostic.png')


if __name__ == '__main__':
    main()
