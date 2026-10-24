#!/usr/bin/env python
"""
PART 1 — Pooled two-house DeepSAD center diagnostic.

Question: after training one encoder on BOTH Haus A and Haus B, does their
NoFlow data form ONE tight cloud around the shared frozen center c, or TWO
separate clouds ("c in no-man's-land")? See the study note:
`_mnt_d_projekte_KILD_inoson_kild_study_center_check_and_noflow_split.txt`.

NOT a leak-detection eval — no GMM, no test data, no leakage patches. Only
NoFlow patches that were part of ENCODER TRAINING, forwarded through the
finished, frozen encoder, in the same z_proj space the DeepSAD center lives
in (L2-normalised projection-head output).

Center source: uses the TRUE frozen training-time center if the checkpoint
was saved after this diagnostic was added (`ckpt['deepsad_center']`). Older
checkpoints (e.g. the current contrastive_g5_deepsad_transformer_ae_hb_diversity
run) don't have it saved, so this script falls back to recomputing the mean
of pooled NoFlow z_proj through the FINAL encoder as a proxy — DeepSAD's loss
already pulls NoFlow toward the true center, so this proxy should sit close
to it, but it is NOT bit-identical and is labelled as such in the output.

Run inside Docker (needs data/prepared/ + a trained checkpoint):
    docker compose run --rm kild python scripts/analyze_deepsad_center_check.py \
        --output-root ./contrastive_g5_deepsad_transformer_ae_hb_diversity

Outputs:
  - data/results/haus_b_adaptation/deepsad_center_check.json
  - data/results/haus_b_adaptation/deepsad_center_check.csv
  - images/contrastive/deepsad_center_check/center_check_pca.png
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.utils import detect_n_stats
from src.contrastive.models import build_contrastive_model
from src.contrastive.train import (
    _load_splits_and_metadata,
    _ids_for_series,
    _prepare_patches,
    _extract_latents,
    _to_channels,
)
from src.contrastive.main import FEATURE_GROUPS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'data' / 'results' / 'haus_b_adaptation'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = ROOT / 'images' / 'contrastive' / 'deepsad_center_check'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Default: matches the ENCODER-TRAIN series actually used for the
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

C_HA = '#2166ac'
C_HB = '#c0392b'
C_CENTER = '#1a1a1a'


def load_checkpoint(output_root: str, device):
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

    with open(results_dir / 'input_scaler.pkl', 'rb') as f:
        scaler_bundle = pickle.load(f)

    return model, ckpt, scaler_bundle, results_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=str, default='./contrastive_g5_deepsad_transformer_ae_hb_diversity')
    parser.add_argument('--data-root', type=str, default='./prepared')
    parser.add_argument('--haus-a-series', nargs='+', default=DEFAULT_HAUS_A)
    parser.add_argument('--haus-b-series', nargs='+', default=DEFAULT_HAUS_B)
    parser.add_argument('--feature-group', type=str, default='g5', choices=list(FEATURE_GROUPS.keys()))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt, scaler_bundle, results_dir = load_checkpoint(args.output_root, device)
    print(f"Loaded checkpoint from {results_dir} (best_epoch={ckpt.get('best_epoch', '?')})")

    data_path = ROOT / 'data'
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, args.data_root)
    # Explicit feature-group selection (NOT ckpt['hp'], which does not store
    # selected_features — see the earlier analyze_haus_ab_feature_distributions.py
    # bug this mirrors: silently loading all 282 columns instead of G5's 18).
    selected = FEATURE_GROUPS[args.feature_group]

    # Step 1: only NoFlow files that were part of encoder training.
    ha_ids = _ids_for_series(splits_data, metadata_by_id, args.haus_a_series)
    hb_ids = _ids_for_series(splits_data, metadata_by_id, args.haus_b_series)
    ha_patches, ha_labels, _, _ = _prepare_patches(ha_ids, prepared_base, metadata_by_id, selected)
    hb_patches, hb_labels, _, _ = _prepare_patches(hb_ids, prepared_base, metadata_by_id, selected)
    ha_noflow = ha_patches[np.asarray(ha_labels) == 0]
    hb_noflow = hb_patches[np.asarray(hb_labels) == 0]
    print(f"Haus A NoFlow (encoder-train) patches: {len(ha_noflow)} | "
          f"Haus B NoFlow (encoder-train) patches: {len(hb_noflow)}")
    if len(ha_noflow) == 0 or len(hb_noflow) == 0:
        raise SystemExit("Need NoFlow patches from both houses in encoder_train_series — check --haus-a/b-series.")

    # Apply the SAME saved (NoFlow-only-fit) scaler + dual-channel reshape the
    # model was trained on — raw unscaled/unshaped patches would silently
    # feed the model out-of-distribution garbage. Mirrors eval_contrastive's
    # exact flow (train.py ~line 2738-2745).
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

    # Step 2+3: forward separately through the final frozen encoder, z_proj space.
    z_a = _extract_latents(model, ha_noflow_c, device, use_proj=True)
    z_b = _extract_latents(model, hb_noflow_c, device, use_proj=True)

    # Center c: true frozen training-time center if saved, else a proxy.
    saved_center = ckpt.get('deepsad_center')
    if saved_center is not None:
        c = saved_center.numpy().astype(np.float64)
        center_source = 'saved_frozen_training_center'
    else:
        z_pooled = np.concatenate([z_a, z_b], axis=0)
        c = z_pooled.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        center_source = ('recomputed_proxy_from_final_encoder — checkpoint predates deepsad_center '
                          'checkpoint field, this is mean(pooled NoFlow z_proj) + L2-normalize using the '
                          'FINAL encoder, not the literal frozen pre-training buffer. DeepSAD pulls NoFlow '
                          'toward the true center during training, so this should sit close to it, but is '
                          'not bit-identical.')
    print(f"Center source: {center_source.split(' — ')[0]}")

    # Step 4: the actual numbers.
    def dist(p, q):
        return float(np.linalg.norm(p - q))

    mean_a = z_a.mean(axis=0)
    mean_b = z_b.mean(axis=0)
    d_a = dist(mean_a, c)
    d_b = dist(mean_b, c)
    spread_a = float(np.linalg.norm(z_a - mean_a, axis=1).mean())
    spread_b = float(np.linalg.norm(z_b - mean_b, axis=1).mean())
    ratio_a = d_a / spread_a if spread_a > 0 else float('nan')
    ratio_b = d_b / spread_b if spread_b > 0 else float('nan')

    flag_two_clouds = (ratio_a > 2.0) or (ratio_b > 2.0)  # d >> spread -> c in no-man's-land

    result = {
        'center_source': center_source,
        'n_haus_a_noflow': int(len(z_a)),
        'n_haus_b_noflow': int(len(z_b)),
        'd_A_to_center': d_a,
        'd_B_to_center': d_b,
        'spread_A': spread_a,
        'spread_B': spread_b,
        'ratio_dA_over_spreadA': ratio_a,
        'ratio_dB_over_spreadB': ratio_b,
        'flag_two_separate_clouds': bool(flag_two_clouds),
        'reading': (
            'ratios small (~<=1-2) => both houses sit tightly around a shared center, one center is fine. '
            'ratios large => two separate blobs, center is in no-man\'s-land, consider one center per house.'
        ),
    }
    print(json.dumps(result, indent=2))

    json_path = OUT_DIR / 'deepsad_center_check.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved -> {json_path}')

    csv_path = OUT_DIR / 'deepsad_center_check.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result.keys())
        writer.writerow([str(v) for v in result.values()])
    print(f'Saved -> {csv_path}')

    # Step 6: PCA-only diagnostic figure (visualization only, NOT the space
    # the reported distances above were computed in).
    z_all = np.concatenate([z_a, z_b, mean_a[None, :], mean_b[None, :], c[None, :]], axis=0)
    pca = PCA(n_components=2).fit(np.concatenate([z_a, z_b], axis=0))
    z_all_2d = pca.transform(z_all)
    n_a, n_b = len(z_a), len(z_b)
    pts_a_2d = z_all_2d[:n_a]
    pts_b_2d = z_all_2d[n_a:n_a + n_b]
    mean_a_2d, mean_b_2d, c_2d = z_all_2d[n_a + n_b], z_all_2d[n_a + n_b + 1], z_all_2d[n_a + n_b + 2]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(pts_a_2d[:, 0], pts_a_2d[:, 1], s=6, alpha=0.35, color=C_HA, label='Haus A NoFlow')
    ax.scatter(pts_b_2d[:, 0], pts_b_2d[:, 1], s=6, alpha=0.35, color=C_HB, label='Haus B NoFlow')
    ax.scatter(*mean_a_2d, s=180, marker='X', color=C_HA, edgecolor='white', linewidth=1.2,
               zorder=5, label='Haus A mean')
    ax.scatter(*mean_b_2d, s=180, marker='X', color=C_HB, edgecolor='white', linewidth=1.2,
               zorder=5, label='Haus B mean')
    ax.scatter(*c_2d, s=260, marker='*', color=C_CENTER, edgecolor='white', linewidth=1.2,
               zorder=6, label='Frozen pooled center c')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.set_title('Pooled Two-House DeepSAD Center Check (Haus A vs. Haus B NoFlow)')
    ax.legend(frameon=False, loc='best', fontsize=9)

    note = (f"PCA is for visualization ONLY (2D projection of a {z_a.shape[1]}-D space). "
            f"All reported distances/ratios are computed in the full latent space, not this projection.\n"
            f"d_A={d_a:.4f} (spread_A={spread_a:.4f}, ratio={ratio_a:.2f})   "
            f"d_B={d_b:.4f} (spread_B={spread_b:.4f}, ratio={ratio_b:.2f})   "
            f"{'FLAG: two separate clouds' if flag_two_clouds else 'one shared cloud looks OK'}")
    fig.text(0.5, 0.01, note, ha='center', va='bottom', fontsize=8, color='#555555', wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = IMG_DIR / 'center_check_pca.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
