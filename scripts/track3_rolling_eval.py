#!/usr/bin/env python
"""
Track 3 — Rolling-Window Haus B Adaptation Evaluation.

Tests whether calibrating the GMM on one hb- series' NoFlow data is enough
to detect leakage in the NEXT hb- series (one week later). This simulates
the realistic deployment scenario: use the most recent confirmed-NoFlow
window as the reference, then score the following measurement period.

Pairs evaluated:
  hb-17.05 NoFlow  →  score hb-25.05  (~1 week gap)
  hb-25.05 NoFlow  →  score hb-31.05  (~1 week gap)

Contrast with:
  Track 2 (per_series): each series calibrates on its own NoFlow → upper bound
  Track 1 (global):     global Haus A GMM → complete failure

Run on the GPU server:
    python scripts/track3_rolling_eval.py
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.contrastive.models import build_contrastive_model
from src.contrastive.train import (
    _load_splits_and_metadata,
    _ids_for_series,
    _prepare_patches,
    _extract_latents,
    _to_channels,
)
from src.common.utils import detect_n_stats

ALLHA_ROOT = 'data/contrastive_g5_deepsad_transformer_ae_allha'
DATA_ROOT = './prepared'

# Rolling calibration pairs: (calibration NoFlow series, eval series)
PAIRS = [
    ('hb-17.05.2026', 'hb-25.05.2026'),
    ('hb-25.05.2026', 'hb-31.05.2026'),
]


def load_model_scaler(root: Path, device: torch.device):
    ckpt_path = root / 'results' / 'contrastive_model.pt'
    scaler_path = root / 'results' / 'input_scaler.pkl'
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
    print(f'  Loaded model  {ckpt_path}  (best_epoch={ckpt.get("best_epoch")})')

    with open(scaler_path, 'rb') as f:
        bundle = pickle.load(f)
    scaler = bundle['scaler']
    feat_names = bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    dual = (bundle.get('cnn_channel_mode', 'dual') == 'dual')
    selected = list(feat_names) if feat_names else None
    return model, scaler, n_stats, dual, selected


def apply_scaler(patches, scaler):
    if len(patches) == 0:
        return patches
    n, f, w = patches.shape
    flat = patches.transpose(0, 2, 1).reshape(-1, f)
    return scaler.transform(flat).reshape(n, w, f).transpose(0, 2, 1).astype(np.float32)


def get_latents(series_name, label_filter, splits_data, metadata_by_id,
                prepared_base, selected, scaler, n_stats, dual, model, device):
    """Extract latents for one series, optionally filtering by label."""
    ids = _ids_for_series(splits_data, metadata_by_id, [series_name])
    if not ids:
        return np.empty((0, 128)), np.empty(0, dtype=int)
    patches, labels, _, _ = _prepare_patches(ids, prepared_base, metadata_by_id, selected)
    if len(patches) == 0:
        return np.empty((0, 128)), np.empty(0, dtype=int)

    labels = np.array(labels)
    if label_filter is not None:
        mask = labels == label_filter
        patches = patches[mask]
        labels = labels[mask]

    if len(patches) == 0:
        return np.empty((0, 128)), np.empty(0, dtype=int)

    patches = apply_scaler(patches, scaler)
    f, w = patches.shape[1], patches.shape[2]
    patches_ch, _ = _to_channels(patches, f, w, n_stats, dual)
    latents = _extract_latents(model, patches_ch, device)
    return latents, labels


def majority_vote(scores, labels, threshold, series_metadata, metadata_by_id, ids):
    """Compute per-file majority vote from patch-level scores."""
    # Map patches back to files via segment IDs
    patch_files = []
    for sid in ids:
        meta = metadata_by_id.get(sid, {})
        n = meta.get('n_patches', 0)
        patch_files.extend([meta.get('file_name', '?')] * n)

    patch_files = np.array(patch_files)
    preds = (scores > threshold).astype(int)  # 1 = leakage
    true_per_file = {}
    pred_per_file = {}

    for fname in sorted(set(patch_files)):
        mask = patch_files == fname
        true_lab = int(labels[mask][0]) if mask.any() else -1
        n_lk = int(preds[mask].sum())
        n_nf = int((~preds[mask].astype(bool)).sum())
        pred_per_file[fname] = 1 if n_lk > n_nf else 0
        true_per_file[fname] = true_lab

    correct = sum(pred_per_file[f] == true_per_file[f] for f in true_per_file)
    return correct, len(true_per_file), true_per_file, pred_per_file


def main():
    import os
    if os.getenv('DOCKER_ENV'):
        root = Path(os.getenv('CONTAINER_APP_PATH', '/app'))
        data_path = Path(os.getenv('CONTAINER_DATA_PATH', '/app/data'))
        img_path = Path(os.getenv('CONTAINER_IMG_PATH', '/app/images'))
    else:
        root = Path(__file__).resolve().parent.parent
        data_path = root / 'data'
        img_path = root / 'images'

    allha_root = data_path / 'contrastive_g5_deepsad_transformer_ae_allha'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model, scaler, n_stats, dual, selected = load_model_scaler(allha_root, device)
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(
        data_path, DATA_ROOT
    )

    print('\n' + '=' * 60)
    print('TRACK 3 — ROLLING WINDOW ADAPTATION')
    print('=' * 60)
    print('Calibrate on series N NoFlow → score series N+1 (~1 week later)')
    print()

    results = {}

    for calib_series, eval_series in PAIRS:
        print(f'\n--- Pair: {calib_series} → {eval_series} ---')

        # Step 1: calibrate GMM on prior series NoFlow
        calib_latents, _ = get_latents(
            calib_series, label_filter=0,
            splits_data=splits_data, metadata_by_id=metadata_by_id,
            prepared_base=prepared_base, selected=selected,
            scaler=scaler, n_stats=n_stats, dual=dual,
            model=model, device=device,
        )
        if len(calib_latents) == 0:
            print(f'  [warn] No NoFlow patches for {calib_series} — skipping.')
            continue

        gmm = GaussianMixture(n_components=1, covariance_type='full',
                              reg_covar=1e-3, random_state=42)
        gmm.fit(calib_latents)
        own_scores = -gmm.score_samples(calib_latents)
        threshold = float(np.percentile(own_scores, 95))
        print(f'  Calibration: {len(calib_latents)} NoFlow patches from {calib_series}')
        print(f'  GMM own score: mean={own_scores.mean():.2f}  threshold (p95)={threshold:.2f}')

        # Step 2: score eval series (all patches, both NoFlow and leakage)
        eval_ids = _ids_for_series(splits_data, metadata_by_id, [eval_series])
        eval_patches, eval_labels, _, _ = _prepare_patches(
            eval_ids, prepared_base, metadata_by_id, selected
        )
        if len(eval_patches) == 0:
            print(f'  [warn] No patches for {eval_series} — skipping.')
            continue

        eval_labels = np.array(eval_labels)
        eval_patches = apply_scaler(eval_patches, scaler)
        f, w = eval_patches.shape[1], eval_patches.shape[2]
        eval_ch, _ = _to_channels(eval_patches, f, w, n_stats, dual)
        eval_latents = _extract_latents(model, eval_ch, device)
        scores = -gmm.score_samples(eval_latents)

        n_noflow = int((eval_labels == 0).sum())
        n_flow = int((eval_labels == 1).sum())
        print(f'  Eval: {len(eval_latents)} patches from {eval_series}  '
              f'(NoFlow={n_noflow}, Flow={n_flow})')

        if n_noflow == 0 or n_flow == 0:
            print('  [warn] Missing one class — cannot compute AUC.')
            continue

        auc = float(roc_auc_score(eval_labels, scores))
        preds = (scores > threshold).astype(int)
        f1 = float(f1_score(eval_labels, preds, zero_division=0))

        # Majority vote
        patch_files = []
        for sid in eval_ids:
            meta = metadata_by_id.get(sid, {})
            n = meta.get('n_patches', 0)
            patch_files.extend([meta.get('file_name', '?')] * n)
        patch_files = np.array(patch_files)

        n_correct, n_total = 0, 0
        file_results = []
        for fname in sorted(set(patch_files)):
            mask = patch_files == fname
            true_lab = int(eval_labels[mask][0])
            n_lk = int(preds[mask].sum())
            n_nf = int((1 - preds[mask]).sum())
            pred = 1 if n_lk > n_nf else 0
            correct = (pred == true_lab)
            n_correct += int(correct)
            n_total += 1
            file_results.append({
                'file': fname, 'true': 'LEAKAGE' if true_lab == 1 else 'NOFLOW',
                'pred': 'LEAKAGE' if pred == 1 else 'NOFLOW',
                'correct': correct,
            })

        print(f'\n  ROC-AUC = {auc:.4f}  |  F1 = {f1:.4f}  |  MV = {n_correct}/{n_total}')
        for r in file_results:
            mark = '✓' if r['correct'] else '✗'
            print(f'    {mark} {r["file"]}:  true={r["true"]}  pred={r["pred"]}')

        results[f'{calib_series}→{eval_series}'] = {
            'calib_series': calib_series,
            'eval_series': eval_series,
            'n_calib_noflow': len(calib_latents),
            'n_eval_noflow': n_noflow,
            'n_eval_flow': n_flow,
            'roc_auc': round(auc, 4),
            'f1': round(f1, 4),
            'majority_vote': f'{n_correct}/{n_total}',
            'threshold': round(threshold, 4),
            'files': file_results,
        }

    # Summary
    print('\n' + '=' * 60)
    print('TRACK 3 SUMMARY')
    print('=' * 60)
    print(f'{"Pair":<40}  {"AUC":>6}  {"F1":>6}  {"MV"}')
    print('-' * 60)
    for key, r in results.items():
        print(f'{key:<40}  {r["roc_auc"]:>6.4f}  {r["f1"]:>6.4f}  {r["majority_vote"]}')

    if results:
        aucs = [r['roc_auc'] for r in results.values()]
        print(f'\n  Mean AUC = {np.mean(aucs):.4f}')

    # Save
    out_path = allha_root / 'results' / 'track3_rolling_eval.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved → {out_path}')


if __name__ == '__main__':
    main()
