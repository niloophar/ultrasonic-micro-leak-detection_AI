"""
Contrastive CNN Encoder Training (Ansatz A).

Trains a contrastive CNN encoder on labelled patches from multiple series
simultaneously. Two variants are selectable via ``encoder_type`` in the
encoder config:

- ``"autoencoder"``: encoder + decoder, loss = MSE(x, x_recon) + λ · SupCon
- ``"classifier"``:  encoder + classifier, loss = CE(logits, y) + λ · SupCon

Both variants use the same CNN encoder backbone, the same L2-normalised
projection head for SupCon, and the same series-stratified batch sampler.
The goal is a temperature-/drift-invariant latent space for one-class
deployment at new sites; downstream GMM scoring uses ``z`` only.

Three entry points used from ``main.py``:

- ``train_contrastive``    — single training run with full diagnostics.
- ``tune_contrastive``     — Optuna study (default 200 trials) with pruning.
- ``loso_contrastive``     — leave-one-series-out evaluation over all series.
"""
import json
import math
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from .losses import SupConLoss, RNCLoss, DeepSADLoss, build_contrastive_criterion
from .models import (
    ContrastiveCNNAutoEncoder,
    ContrastiveCNNClassifier,
    build_contrastive_model,
)
from src.anomaly_detection.train import _load_prepared_data
from src.common.utils import detect_n_stats, reshape_to_channels


def _is_autoencoder(encoder_type: str) -> bool:
    return encoder_type.lower() in ('autoencoder', 'ae')


# Keys that define which variant was trained (loss recipe + aux head + data
# shape). These travel in `optuna_best_params.json -> "variant"` so a later
# `train` / `loso` reproduces the tuned recipe even when the JSON default has
# drifted. `cnn_channel_mode` and `encoder_arch` are included because
# they change the input tensor shape — a tuned model cannot be reused with
# different values without retraining.
_VARIANT_KEYS = (
    'encoder_type', 'encoder_arch', 'contrastive_loss', 'rnc_target', 'rnc_similarity',
    'cnn_channel_mode',
)


def _extract_variant_from_cfg(cfg: dict) -> dict:
    """Snapshot the variant-defining fields currently active on cfg."""
    return {k: cfg.get(k) for k in _VARIANT_KEYS if cfg.get(k) is not None}


def _apply_variant_from_best_params(
    cfg: dict, best_payload: dict,
) -> None:
    """Sync cfg with the variant block recorded at tune-time.

    Precedence: CLI override (recorded in cfg['_cli_overridden_keys']) beats the
    variant block, which beats the JSON config default. Mismatches are printed
    so the user sees exactly what changed. If no variant block is present
    (older tune artefact), warn — the silent fallback to JSON default is
    exactly the bug this helper exists to prevent.
    """
    variant = best_payload.get('variant') or {}
    cli_overridden = set(cfg.get('_cli_overridden_keys', []))

    if not variant:
        print(
            "  [variant] optuna_best_params.json has no variant block — "
            "cannot verify which loss/aux recipe was tuned. "
            "Check --contrastive-loss / --encoder-type manually, or re-run tune "
            "to refresh the file."
        )
        return

    for key in _VARIANT_KEYS:
        if key not in variant or variant[key] is None:
            continue
        stored = variant[key]
        current = cfg.get(key)
        if key in cli_overridden:
            if current != stored:
                print(
                    f"  [variant] {key}={stored!r} from tune — kept CLI override "
                    f"{current!r}"
                )
            continue
        if current != stored:
            print(f"  [variant] {key}: {current!r} (cfg) → {stored!r} (from tune)")
        cfg[key] = stored


def _aux_loss_label(encoder_type: str) -> str:
    """Short label used in the training log (mse for AE, ce for classifier)."""
    return 'mse' if _is_autoencoder(encoder_type) else 'ce'


def _resolve_selected_features(cfg: dict) -> Optional[List[str]]:
    """Return the configured ``selected_features`` list, or None to load all."""
    selected = cfg.get('selected_features')
    return list(selected) if selected else selected


# ===========================================================================
# Dataset + series-stratified batch sampler
# ===========================================================================

class PatchDataset(Dataset):
    """In-memory patch dataset with labels, series, and an optional
    continuous per-patch target used by :class:`RNCLoss`.

    The continuous target is zero-initialised when not provided so that
    downstream code can always index ``item[2]`` without branching.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        series: List[str],
        regress_target: Optional[np.ndarray] = None,
    ):
        self.patches = torch.tensor(patches, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.series = list(series)
        if regress_target is not None:
            self.regress_target = torch.tensor(regress_target, dtype=torch.float32)
        else:
            self.regress_target = torch.zeros(len(patches), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        return (
            self.patches[idx],
            self.labels[idx],
            self.regress_target[idx],
            idx,
        )


class SeriesStratifiedBatchSampler(Sampler[List[int]]):
    """
    Yields batches with ``k_per_series`` patches per series, mixed labels.

    Forces every training batch to contain samples from **all** available
    series — this is what lets SupConLoss learn cross-series invariance.
    Random shuffling alone tends to give batches dominated by a single
    series when segment counts are uneven.

    Within each series the sampler draws with replacement when the series
    runs out of patches, so batches stay the same size for the whole epoch.
    ``__len__`` is an estimate based on the largest-series exhaustion.
    """

    def __init__(
        self,
        series_per_sample: List[str],
        k_per_series: int,
        seed: int = 42,
    ):
        self.series_per_sample = list(series_per_sample)
        self.k_per_series = k_per_series
        self.rng = random.Random(seed)

        self._indices_by_series: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(self.series_per_sample):
            self._indices_by_series[s].append(i)

        self._series_list = sorted(self._indices_by_series.keys())
        self._n_series = len(self._series_list)

        self._batch_size = self._n_series * self.k_per_series

        # Epoch length = largest series / k_per_series (rounded up)
        largest = max(len(v) for v in self._indices_by_series.values())
        self._num_batches = max(1, math.ceil(largest / self.k_per_series))

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self):
        # Shuffle each series' indices at the start of every epoch.
        pools: Dict[str, List[int]] = {}
        for s, idxs in self._indices_by_series.items():
            pool = list(idxs)
            self.rng.shuffle(pool)
            pools[s] = pool

        cursors = {s: 0 for s in self._series_list}

        for _ in range(self._num_batches):
            batch: List[int] = []
            for s in self._series_list:
                pool = pools[s]
                take = self.k_per_series
                available = len(pool) - cursors[s]
                if available >= take:
                    batch.extend(pool[cursors[s]:cursors[s] + take])
                    cursors[s] += take
                else:
                    # Take what's left then reshuffle and continue (with replacement).
                    if available > 0:
                        batch.extend(pool[cursors[s]:])
                        take -= available
                    self.rng.shuffle(pool)
                    cursors[s] = take
                    batch.extend(pool[:take])
            yield batch

    def __len__(self) -> int:
        return self._num_batches


# ===========================================================================
# Shared data-loading helper
# ===========================================================================

def _load_splits_and_metadata(
    data_path: Path, data_root: str
) -> Tuple[dict, Dict[int, dict], Path]:
    class_base = data_path / data_root
    splits_path = class_base / 'splits.json'
    prep_meta_path = class_base / 'prepared' / 'metadata.json'

    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_path}")
    if not prep_meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {prep_meta_path}")

    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    with open(prep_meta_path, 'r') as f:
        all_meta = json.load(f)

    # Contrastive fits its own NoFlow-only StandardScaler on top of the loaded
    # patches to match Scaler A semantics. That only produces the intended
    # result if the patches on disk are raw (scaling_mode="none"). A prep run
    # with noflow_global would double-standardize (harmless), but
    # segment_center / time_diff / noflow_per_series would silently give
    # garbage. Fail loud instead.
    modes = {m.get('prep_scaling_mode', 'none') for m in all_meta}
    if modes - {'none'}:
        raise RuntimeError(
            f"Contrastive pipeline requires patches prepared with "
            f"scaling_mode='none', but metadata reports {sorted(modes)}. "
            f"Re-run `python -m src.preparation.main --run prepare` with "
            f"data_preparation.json -> scaling_mode='none'."
        )

    metadata_by_id = {m['segment_id']: m for m in all_meta}
    return splits_data, metadata_by_id, class_base / 'prepared'


def _ids_for_series(
    splits_data: dict,
    metadata_by_id: Dict[int, dict],
    allowed_series: List[str],
) -> List[int]:
    """All segment IDs (train+val+test) whose series is in ``allowed_series``."""
    all_ids: List[int] = []
    for split in ('train', 'val', 'test'):
        all_ids.extend(splits_data['splits'].get(split, []))
    allowed = set(allowed_series)
    return [
        sid for sid in all_ids
        if metadata_by_id.get(sid, {}).get('series') in allowed
    ]


def _ids_for_blind_series(
    splits_data: dict,
    metadata_by_id: Dict[int, dict],
    allowed_series: List[str],
) -> List[int]:
    """Segment IDs from the blind_test bucket whose series is in ``allowed_series``.
    Falls back to all splits (train+val+test) for --as-blind-series usage where
    the series has true labels but is intentionally treated as unlabelled."""
    blind_ids: List[int] = list(splits_data['splits'].get('blind_test', []))
    allowed = set(allowed_series)
    matched = [sid for sid in blind_ids if metadata_by_id.get(sid, {}).get('series') in allowed]
    if matched:
        return matched
    # Fallback: series has labels (--as-blind-series sanity mode)
    return _ids_for_series(splits_data, metadata_by_id, allowed_series)


def _segment_train_val_split(
    segment_ids: List[int],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministic segment-level 80/20 split. Used when encoder_train_series
    is overridden and we cannot reuse splits.json's pre-computed train/val
    partition."""
    if not segment_ids:
        return [], []
    sorted_ids = sorted(segment_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sorted_ids))
    n_val = max(1, int(round(len(sorted_ids) * val_fraction)))
    val_idx = set(perm[:n_val].tolist())
    val_ids = [sorted_ids[i] for i in range(len(sorted_ids)) if i in val_idx]
    train_ids = [sorted_ids[i] for i in range(len(sorted_ids)) if i not in val_idx]
    return train_ids, val_ids


def _resolve_train_eval_ids(
    splits_data: dict,
    metadata_by_id: Dict[int, dict],
    encoder_train_series: Optional[List[str]],
    gmm_eval_series: Optional[List[str]],
) -> Tuple[List[int], List[int], List[int], List[str], List[str]]:
    """
    Decide which segment IDs feed (a) encoder train+val and (b) the eval set,
    based on optional series overrides.

    Behaviour:
    - ``encoder_train_series=None`` and matches the default → reuse
      ``splits.json``'s precomputed ``train`` and ``val`` lists verbatim
      (back-compat with all prior runs).
    - Otherwise → gather every segment ID from the requested series across
      all of train/val/test in splits.json and re-split 80/20 segment-wise.
    - ``gmm_eval_series=None`` → use ``splits.json['splits']['test']`` (the
      default Spring-5 hold-out).
    - Otherwise → gather every segment ID from the requested series.

    Returns ``(train_ids, val_ids, test_ids, encoder_train_series_resolved,
    gmm_eval_series_resolved)`` so the caller can log what it actually ran.
    """
    default_train_series = sorted(splits_data.get('train_series', []))
    default_test_series = sorted(splits_data.get('test_series', []))

    # Encoder train/val
    if encoder_train_series is None:
        train_ids = list(splits_data['splits'].get('train', []))
        val_ids = list(splits_data['splits'].get('val', []))
        train_resolved = default_train_series
    elif sorted(encoder_train_series) == default_train_series:
        # Equivalent to default — preserve splits.json partitioning exactly.
        train_ids = list(splits_data['splits'].get('train', []))
        val_ids = list(splits_data['splits'].get('val', []))
        train_resolved = sorted(encoder_train_series)
    else:
        all_ids = _ids_for_series(splits_data, metadata_by_id, encoder_train_series)
        train_ids, val_ids = _segment_train_val_split(all_ids)
        train_resolved = sorted(encoder_train_series)

    # GMM eval set
    if gmm_eval_series is None:
        test_ids = list(splits_data['splits'].get('test', []))
        eval_resolved = default_test_series
    elif sorted(gmm_eval_series) == default_test_series:
        test_ids = list(splits_data['splits'].get('test', []))
        eval_resolved = sorted(gmm_eval_series)
    else:
        test_ids = _ids_for_series(splits_data, metadata_by_id, gmm_eval_series)
        eval_resolved = sorted(gmm_eval_series)

    # Sanity: a series should not appear in both train and eval — would mean
    # the encoder saw the held-out NoFlow during training and the eval is
    # leaky. Warn but don't abort (LOSO-style runs may want this).
    overlap = set(train_resolved) & set(eval_resolved)
    if overlap:
        print(
            f"  [warn] series overlap between encoder_train and gmm_eval: "
            f"{sorted(overlap)} — eval will not be held-out for these."
        )

    return train_ids, val_ids, test_ids, train_resolved, eval_resolved


def _prepare_patches(
    segment_ids: List[int],
    prepared_base: Path,
    metadata_by_id: Dict[int, dict],
    selected_features: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[str]]]:
    """Wrapper around AD's ``_load_prepared_data`` that preserves label+series."""
    patches, labels, series, feat_names = _load_prepared_data(
        segment_ids, prepared_base, metadata_by_id,
        selected_features=selected_features,
        label_filter=None,  # keep both NoFlow and Flow
    )
    return patches, labels, series, feat_names


def _scale_patches(
    train_patches: np.ndarray,
    train_labels: np.ndarray,
    other_sets: List[np.ndarray],
    scaler_type: str,
) -> Tuple[List[np.ndarray], object]:
    """
    Fit scaler on train-NoFlow only, transform all sets.

    NoFlow-only fitting matches the prep-time "Scaler A" (noflow_global) data
    flow: the input distribution seen by the encoder is identical to the
    SCALER_A CNN baseline (so SupCon is a fair ablation on top of the same
    data), while the drift invariance itself remains to be learned by the
    contrastive objective rather than baked into the scaler.
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    noflow_mask = np.asarray(train_labels) == 0
    if noflow_mask.sum() == 0:
        raise RuntimeError("Cannot fit NoFlow-only scaler: no NoFlow training patches.")

    fit_patches = train_patches[noflow_mask]
    _n, n_feat, _w = fit_patches.shape
    flat = fit_patches.transpose(0, 2, 1).reshape(-1, n_feat)
    scaler.fit(flat)

    def _apply(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return arr
        n_, f_, w_ = arr.shape
        flat_ = arr.transpose(0, 2, 1).reshape(-1, f_)
        return scaler.transform(flat_).reshape(n_, w_, f_).transpose(0, 2, 1)

    return [_apply(train_patches)] + [_apply(a) for a in other_sets], scaler


def _to_channels(
    arr: np.ndarray, n_feat: int, w: int, n_stats: int, dual: bool
) -> Tuple[np.ndarray, int]:
    if len(arr) == 0:
        if dual and n_feat % 2 == 0:
            n_channels = 2 * n_stats
            return arr.reshape(0, n_channels, n_feat // n_stats // 2, w), n_channels
        return arr.reshape(0, 1, n_feat, w), 1
    if dual and n_feat % 2 == 0:
        reshaped, n_in_channels, _ = reshape_to_channels(arr, n_feat, w, n_stats)
        return reshaped, n_in_channels
    return arr[:, np.newaxis, :, :], 1


# ===========================================================================
# DeepSAD center initialization
# ===========================================================================

def _init_deepsad_center(
    model: nn.Module,
    train_patches: np.ndarray,
    train_labels: np.ndarray,
    deepsad_criterion: 'DeepSADLoss',
    device: torch.device,
    batch_size: int = 256,
) -> int:
    """Initialize DeepSAD center from ALL train-NoFlow projections.

    Direct numpy-array iteration (no sampler, no replacement) guarantees
    exhaustive, deterministic coverage of every NoFlow patch. Returns the
    number of NoFlow patches used so the caller can log it.
    """
    model.eval()
    noflow_mask = np.asarray(train_labels) == 0
    if not noflow_mask.any():
        raise RuntimeError("_init_deepsad_center: no NoFlow patches in train set.")

    patches_nf = torch.tensor(np.asarray(train_patches)[noflow_mask], dtype=torch.float32)
    z_proj_chunks = []

    with torch.no_grad():
        for i in range(0, len(patches_nf), batch_size):
            chunk = patches_nf[i:i + batch_size].to(device)
            _aux_out, _z, z_proj = model(chunk)
            z_proj_chunks.append(z_proj.detach())

    all_z_proj = torch.cat(z_proj_chunks, dim=0).to(device)
    deepsad_criterion.set_center(all_z_proj)
    model.train()
    return int(noflow_mask.sum())


# ===========================================================================
# Training loop
# ===========================================================================

def _compute_aux_loss(
    aux_output: torch.Tensor,
    patches: torch.Tensor,
    labels: torch.Tensor,
    aux_criterion: nn.Module,
    encoder_type: str,
) -> torch.Tensor:
    """Compute auxiliary loss depending on encoder variant.

    AE: MSE(x_recon, patches). Classifier: CE(logits, labels).
    """
    if _is_autoencoder(encoder_type):
        return aux_criterion(aux_output, patches)
    return aux_criterion(aux_output, labels)


def _contrastive_step(
    z_proj: torch.Tensor,
    labels: torch.Tensor,
    regress_target: torch.Tensor,
    criteria: Dict,
    lambda_supcon: float,
    lambda_rnc: float,
    lambda_deepsad: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the contrastive loss component(s) from the configured criteria.

    Returns (weighted_total, raw_supcon, raw_rnc, raw_deepsad) where missing
    components are zero tensors. Keeping the raw values separate lets the
    training log show the λ tradeoff for each active loss.
    """
    device = z_proj.device
    zero = torch.zeros((), device=device)
    supcon_val = zero
    rnc_val = zero
    deepsad_val = zero
    total = zero

    if criteria.get('supcon') is not None:
        supcon_val = criteria['supcon'](z_proj, labels)
        total = total + lambda_supcon * supcon_val
    if criteria.get('rnc') is not None:
        rnc_val = criteria['rnc'](z_proj, regress_target)
        total = total + lambda_rnc * rnc_val
    if criteria.get('deepsad') is not None:
        deepsad_val = criteria['deepsad'](z_proj, labels)
        total = total + lambda_deepsad * deepsad_val
    return total, supcon_val, rnc_val, deepsad_val


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    aux_criterion: nn.Module,
    criteria: Dict,
    lambda_supcon: float,
    lambda_rnc: float,
    lambda_deepsad: float,
    device: torch.device,
    encoder_type: str,
) -> Tuple[float, float, float, float, float]:
    model.train()
    tot_aux, tot_sc, tot_rn, tot_ds, tot_total, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for patches, labels, regress_target, _idx in loader:
        patches = patches.to(device)
        labels = labels.to(device)
        regress_target = regress_target.to(device)
        optimizer.zero_grad()

        aux_out, _z, z_proj = model(patches)
        aux = _compute_aux_loss(aux_out, patches, labels, aux_criterion, encoder_type)
        contrastive_weighted, supcon_val, rnc_val, deepsad_val = _contrastive_step(
            z_proj, labels, regress_target, criteria,
            lambda_supcon, lambda_rnc, lambda_deepsad,
        )
        total = aux + contrastive_weighted

        total.backward()
        optimizer.step()

        tot_aux += aux.item()
        tot_sc += supcon_val.item()
        tot_rn += rnc_val.item()
        tot_ds += deepsad_val.item()
        tot_total += total.item()
        n_batches += 1

    n = max(n_batches, 1)
    return tot_aux / n, tot_sc / n, tot_rn / n, tot_ds / n, tot_total / n


def _eval_losses(
    model: nn.Module,
    loader: DataLoader,
    aux_criterion: nn.Module,
    criteria: Dict,
    lambda_supcon: float,
    lambda_rnc: float,
    lambda_deepsad: float,
    device: torch.device,
    encoder_type: str,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    tot_aux, tot_sc, tot_rn, tot_ds, tot_total, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for patches, labels, regress_target, _idx in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            regress_target = regress_target.to(device)
            aux_out, _z, z_proj = model(patches)
            aux = _compute_aux_loss(aux_out, patches, labels, aux_criterion, encoder_type)
            contrastive_weighted, supcon_val, rnc_val, deepsad_val = _contrastive_step(
                z_proj, labels, regress_target, criteria,
                lambda_supcon, lambda_rnc, lambda_deepsad,
            )
            total = aux + contrastive_weighted
            tot_aux += aux.item()
            tot_sc += supcon_val.item()
            tot_rn += rnc_val.item()
            tot_ds += deepsad_val.item()
            tot_total += total.item()
            n += 1
    n = max(n, 1)
    return tot_aux / n, tot_sc / n, tot_rn / n, tot_ds / n, tot_total / n


def _extract_latents(
    model: nn.Module,
    patches: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    use_proj: bool = False,
) -> np.ndarray:
    """Extract z (encoder output) or z_proj (L2-normalised projection head output)."""
    model.eval()
    out = []
    t_patches = torch.tensor(patches, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(t_patches), batch_size):
            chunk = t_patches[i:i + batch_size].to(device)
            z = model.encode(chunk)
            if use_proj:
                z = model.projection(z)
            out.append(z.cpu().numpy())
    if not out:
        dim = model.projection_dim if use_proj else model.latent_dim
        return np.empty((0, dim))
    return np.concatenate(out, axis=0)


# ===========================================================================
# Evaluation: GMM on NoFlow latents (deployment simulation)
# ===========================================================================

def _gmm_score(
    train_noflow_latents: np.ndarray,
    eval_latents: np.ndarray,
    n_components: int = 1,
    covariance_type: str = 'full',
) -> np.ndarray:
    """Fit GMM on held-out NoFlow latents and return negative log-likelihood as anomaly score.

    May raise ``np.linalg.LinAlgError`` / ``ValueError`` when the empirical NoFlow
    covariance is rank-deficient (SupCon-driven latent collapse at high latent_dim).
    Callers should handle the failure — :func:`_evaluate_on_test` skips the
    affected series instead of aborting the whole Optuna trial.
    """
    # reg_covar=1e-3 is a soft diagonal loading to tolerate mild rank deficiency.
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=1e-3,
        random_state=42,
    )
    gmm.fit(train_noflow_latents)
    # score_samples returns log p(x) — higher = more normal. Negate for anomaly score.
    return -gmm.score_samples(eval_latents)


def _auc_safe(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) < 2:
            return 0.5
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _f1_with_gauss_threshold(
    noflow_scores: np.ndarray,
    all_labels: np.ndarray,
    all_scores: np.ndarray,
    sigma_k: float = 3.0,
) -> Tuple[float, float]:
    """
    Adaptive Gaussian threshold from calibration-NoFlow scores.
    threshold = mu + sigma_k * sigma.
    Returns (f1, threshold).
    """
    mu = float(noflow_scores.mean())
    sigma = float(noflow_scores.std() + 1e-12)
    threshold = mu + sigma_k * sigma

    pred = (all_scores > threshold).astype(int)
    tp = int(((pred == 1) & (all_labels == 1)).sum())
    fp = int(((pred == 1) & (all_labels == 0)).sum())
    fn = int(((pred == 0) & (all_labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1, threshold


def _classification_metrics_at_threshold(
    all_labels: np.ndarray,
    all_scores: np.ndarray,
    threshold: float,
) -> Dict:
    """
    Patch-level accuracy, correct/wrong counts, and confusion matrix at a
    fixed anomaly-score threshold. Uses the SAME prediction rule as
    `_f1_with_gauss_threshold` (score > threshold → predicted leakage=1),
    so accuracy/confusion-matrix here are consistent with the F1 already
    reported alongside them — call with the same `threshold` value.

    Added on top of the existing roc_auc/f1/threshold/n_noflow/n_flow
    metrics (does not replace or recompute any of them).
    """
    labels = np.asarray(all_labels).astype(int)
    pred = (np.asarray(all_scores) > threshold).astype(int)

    total = int(len(labels))
    n_correct = int((pred == labels).sum())
    n_wrong = total - n_correct
    accuracy = n_correct / total if total > 0 else 0.0

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    return {
        'accuracy': accuracy,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }


def _aggregate_classification_metrics(out: Dict[str, Dict]) -> Dict:
    """
    Sums the per-series accuracy/n_correct/n_wrong/confusion_matrix entries
    produced by `_classification_metrics_at_threshold` into one summary dict
    — mean_accuracy, total_n_correct, total_n_wrong, and a pooled confusion
    matrix. Skips the 'overall' key itself and any series without these
    fields (e.g. skipped/undefined series). Merge the result into an
    existing 'overall' dict; does not touch mean_roc_auc/mean_f1/n_series.
    """
    series_entries = [
        v for k, v in out.items() if k != 'overall' and 'accuracy' in v
    ]
    if not series_entries:
        return {}
    cm = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for v in series_entries:
        for key in cm:
            cm[key] += v['confusion_matrix'][key]
    return {
        'mean_accuracy': float(np.mean([v['accuracy'] for v in series_entries])),
        'total_n_correct': sum(v['n_correct'] for v in series_entries),
        'total_n_wrong': sum(v['n_wrong'] for v in series_entries),
        'confusion_matrix': cm,
    }


# ===========================================================================
# Diagnostics plotting
# ===========================================================================

def _plot_loss_curves(
    history: Dict[str, List[float]], save_path: Path
):
    aux_label = history.get('aux_label', 'aux')
    panels = [('aux', aux_label.upper()), ('supcon', 'SUPCON'), ('rnc', 'RNC'), ('deepsad', 'DEEPSAD'), ('total', 'TOTAL')]
    # Hide any panel that recorded only zeros (the loss wasn't active).
    active = []
    for key, title in panels:
        if f'train_{key}' not in history:
            continue
        train_vals = history[f'train_{key}']
        val_vals = history.get(f'val_{key}', [])
        if not train_vals:
            continue
        if all(v == 0.0 for v in train_vals) and all(v == 0.0 for v in val_vals):
            continue
        active.append((key, title))

    if not active:
        return
    fig, axes = plt.subplots(1, len(active), figsize=(6 * len(active), 5), squeeze=False)
    for ax, (key, title) in zip(axes[0], active):
        ax.plot(history[f'train_{key}'], label='train')
        ax.plot(history[f'val_{key}'], label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_reconstructions(
    model: ContrastiveCNNAutoEncoder,
    patches: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    save_path: Path,
    n_per_class: int = 3,
):
    """Only valid for the autoencoder variant (produces x_recon)."""
    model.eval()
    noflow_idx = np.where(labels == 0)[0][:n_per_class]
    flow_idx = np.where(labels == 1)[0][:n_per_class]
    chosen = np.concatenate([noflow_idx, flow_idx])
    if len(chosen) == 0:
        return

    with torch.no_grad():
        t = torch.tensor(patches[chosen], dtype=torch.float32).to(device)
        recon, _z, _zp = model(t)
        recon = recon.cpu().numpy()

    original = patches[chosen]

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)
    for i, idx in enumerate(chosen):
        lbl = 'NoFlow' if labels[idx] == 0 else 'Flow'
        # Collapse channel dim by averaging for display.
        orig_img = original[i].mean(axis=0) if original[i].ndim == 3 else original[i]
        rec_img = recon[i].mean(axis=0) if recon[i].ndim == 3 else recon[i]
        axes[i, 0].imshow(orig_img, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f'{lbl} — original')
        axes[i, 1].imshow(rec_img, aspect='auto', cmap='viridis')
        axes[i, 1].set_title(f'{lbl} — reconstruction')
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_latent_tsne(
    latents: np.ndarray,
    labels: np.ndarray,
    series: List[str],
    save_path_label: Path,
    save_path_series: Path,
):
    if len(latents) < 5:
        return
    perplexity = min(30, max(5, len(latents) // 4))
    coords = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(latents)

    # By label
    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl, color, name in [(0, 'steelblue', 'NoFlow'), (1, 'salmon', 'Flow')]:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.5,
                   c=color, label=f'{name} (n={mask.sum()})')
    ax.set_title('Contrastive Latent Space — by label (t-SNE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path_label, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path_label}")

    # By series
    series_arr = np.array(series)
    unique = sorted(set(series))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, s in enumerate(unique):
        mask = series_arr == s
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, alpha=0.5,
                   c=[colors[i]], label=f'{s} (n={mask.sum()})')
    ax.set_title('Contrastive Latent Space — by series (t-SNE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path_series, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path_series}")


def _noflow_compactness_per_series(
    latents: np.ndarray,
    labels: np.ndarray,
    series: List[str],
) -> Dict[str, float]:
    """Mean Euclidean distance of NoFlow latents to the series centroid."""
    series_arr = np.array(series)
    result: Dict[str, float] = {}
    for s in sorted(set(series)):
        mask = (series_arr == s) & (labels == 0)
        if mask.sum() < 2:
            result[s] = float('nan')
            continue
        pts = latents[mask]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        result[s] = float(dists.mean())
    return result


# ===========================================================================
# Core single-run training (shared by train_contrastive / tune / loso)
# ===========================================================================

def _compute_per_series_regress_target(
    data_path: Path,
    series_names: List[str],
    target_name: str,
) -> Dict[str, float]:
    """Per-series continuous target for RNC loss, pulled from processed CSVs.

    Currently supports ``temp_pipe``, ``temp_amb``, ``temp_diff`` (means over
    NoFlow + equilibrium + valid rows of ``data/processed/<series>/*.csv``)
    and ``series_index`` (fallback when CSVs aren't available).

    Returns a dict ``{series_name: value}``. Series with no available data
    receive their ordinal index so that RNC still has a valid ordering.
    """
    processed_root = data_path / 'processed'
    PIPE_COL = "Temp Pipe in °C"
    AMB_COL = "Temp Ambient in °C"

    target_name = target_name.lower()
    if target_name == 'series_index':
        return {s: float(i) for i, s in enumerate(series_names)}

    out: Dict[str, float] = {}
    for i, s in enumerate(series_names):
        series_dir = processed_root / s
        if not series_dir.exists():
            out[s] = float(i)  # fallback
            continue
        pipe_vals, amb_vals = [], []
        for csv_file in sorted(series_dir.glob('*.csv')):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
            except Exception:
                continue
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
            out[s] = float(i)
            continue
        pipe_arr = np.array(pipe_vals)
        amb_arr = np.array(amb_vals)
        if target_name == 'temp_pipe':
            out[s] = float(pipe_arr.mean())
        elif target_name == 'temp_amb':
            out[s] = float(amb_arr.mean())
        elif target_name == 'temp_diff':
            out[s] = float((pipe_arr - amb_arr).mean())
        else:
            raise ValueError(
                f"Unknown rnc_target {target_name!r}. "
                "Use 'temp_pipe', 'temp_amb', 'temp_diff', or 'series_index'."
            )
    return out


def _series_list_to_targets(
    series: List[str], mapping: Dict[str, float]
) -> np.ndarray:
    """Vectorise a per-series target lookup over a patch-level series list."""
    return np.array([mapping.get(s, 0.0) for s in series], dtype=np.float32)


def _regress_target_map_if_needed(
    data_path: Path,
    cfg: dict,
    metadata_by_id: Dict[int, dict],
) -> Optional[Dict[str, float]]:
    """Build the per-series continuous target only when RNC is active.

    Reading processed CSVs is expensive, so we skip it when the configured
    ``contrastive_loss`` is pure SupCon (or disabled). The caller passes
    the returned mapping (or ``None``) into :func:`_prepare_dataset_tensors`.
    """
    contrastive_loss = str(cfg.get('contrastive_loss', 'supcon')).lower()
    if contrastive_loss not in ('rnc', 'both', 'supcon+rnc', 'rnc+supcon'):
        return None
    target_name = cfg.get('rnc_target', 'temp_pipe')
    unique_series = sorted({m['series'] for m in metadata_by_id.values()})
    mapping = _compute_per_series_regress_target(data_path, unique_series, target_name)
    print(f"RNC target = {target_name}. Per-series values:")
    for s in unique_series:
        print(f"  {s}: {mapping.get(s, float('nan')):.4f}")
    return mapping


def _prepare_dataset_tensors(
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    prepared_base: Path,
    metadata_by_id: Dict[int, dict],
    selected_features: Optional[List[str]],
    scaler_type: str,
    dual: bool,
    regress_target_map: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Load, scale, and reshape patches for a (train_ids, val_ids, test_ids) split.
    Returns a dict with all tensors + metadata needed for training+eval.
    """
    print("  Loading train patches...")
    tr_p, tr_l, tr_s, feat_names = _prepare_patches(
        train_ids, prepared_base, metadata_by_id, selected_features
    )
    print(f"    → {tr_p.shape}, NoFlow={(tr_l == 0).sum()}, Flow={(tr_l == 1).sum()}")

    print("  Loading val patches...")
    va_p, va_l, va_s, _ = _prepare_patches(
        val_ids, prepared_base, metadata_by_id, selected_features
    )
    print(f"    → {va_p.shape}, NoFlow={(va_l == 0).sum()}, Flow={(va_l == 1).sum()}")

    print("  Loading test patches...")
    te_p, te_l, te_s, _ = _prepare_patches(
        test_ids, prepared_base, metadata_by_id, selected_features
    )
    print(f"    → {te_p.shape}, NoFlow={(te_l == 0).sum()}, Flow={(te_l == 1).sum()}")

    if len(tr_p) == 0:
        raise RuntimeError("No training patches loaded.")

    (tr_s_p, va_s_p, te_s_p), scaler = _scale_patches(
        tr_p, tr_l, [va_p, te_p], scaler_type,
    )

    _, n_feat, w = tr_p.shape
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    tr_c, n_in = _to_channels(tr_s_p, n_feat, w, n_stats, dual)
    va_c, _ = _to_channels(va_s_p, n_feat, w, n_stats, dual)
    te_c, _ = _to_channels(te_s_p, n_feat, w, n_stats, dual)

    # Per-patch continuous target (for RNC). Defaults to zero — training
    # code only consults this tensor when the RNC criterion is active.
    if regress_target_map is not None:
        tr_t = _series_list_to_targets(tr_s, regress_target_map)
        va_t = _series_list_to_targets(va_s, regress_target_map)
        te_t = _series_list_to_targets(te_s, regress_target_map)
    else:
        tr_t = np.zeros(len(tr_s), dtype=np.float32)
        va_t = np.zeros(len(va_s), dtype=np.float32)
        te_t = np.zeros(len(te_s), dtype=np.float32)

    return {
        'train_patches': tr_c, 'train_labels': tr_l, 'train_series': tr_s,
        'train_target': tr_t,
        'val_patches': va_c, 'val_labels': va_l, 'val_series': va_s,
        'val_target': va_t,
        'test_patches': te_c, 'test_labels': te_l, 'test_series': te_s,
        'test_target': te_t,
        'feature_names': feat_names,
        'n_in_channels': n_in,
        'input_height': tr_c.shape[2],
        'input_width': tr_c.shape[3],
        'scaler': scaler,
    }


def _build_loaders(
    data: Dict,
    k_per_series: int,
    val_batch_size: int,
) -> Tuple[DataLoader, Optional[DataLoader], int]:
    train_ds = PatchDataset(
        data['train_patches'], data['train_labels'], data['train_series'],
        regress_target=data.get('train_target'),
    )
    sampler = SeriesStratifiedBatchSampler(
        series_per_sample=data['train_series'],
        k_per_series=k_per_series,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if len(data['val_patches']) > 0:
        val_ds = PatchDataset(
            data['val_patches'], data['val_labels'], data['val_series'],
            regress_target=data.get('val_target'),
        )
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, sampler.batch_size


def _run_single_training(
    data: Dict,
    hp: Dict,
    device: torch.device,
    epochs: int,
    patience: int,
    optuna_trial=None,
    verbose: bool = True,
    center_out: Optional[dict] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], int]:
    """
    Run one train loop. Used by both train_contrastive and tune_contrastive.
    Returns (model, history, best_epoch).

    Branches on ``hp['encoder_type']`` ('autoencoder' vs 'classifier'). The
    auxiliary loss is MSE for AE, cross-entropy for Classifier; SupCon is
    identical in both cases.

    ``center_out``: optional dict, populated in-place with the frozen DeepSAD
    center (``center_out['deepsad_center'] = tensor``) if the run uses DeepSAD
    loss. Return signature is unchanged (backward compatible with every
    existing caller) — this is opt-in only, callers that don't pass it see no
    behaviour change. Needed so the checkpoint can persist the TRUE frozen
    training-time center rather than a later approximation.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    encoder_type = hp.get('encoder_type', 'autoencoder')
    aux_label = _aux_loss_label(encoder_type)

    train_loader, val_loader, batch_size = _build_loaders(
        data, hp['k_per_series'], val_batch_size=max(32, hp['k_per_series'] * 4)
    )

    model = build_contrastive_model(
        encoder_type=encoder_type,
        encoder_arch=hp.get('encoder_arch', 'cnn'),
        input_height=data['input_height'],
        input_width=data['input_width'],
        channels=hp.get('cnn_channels') or [32, 64, 128],
        kernel_size=hp.get('cnn_kernel_size') or 3,
        latent_dim=hp['latent_dim'],
        projection_dim=hp['projection_dim'],
        dropout=hp.get('dropout', 0.3),
        in_channels=data['n_in_channels'],
        d_model=hp.get('d_model', 128),
        num_heads=hp.get('num_heads', 4),
        num_layers=hp.get('num_layers', 4),
        ff_dim=hp.get('ff_dim', 256),
        transformer_dropout=hp.get('transformer_dropout', 0.1),
    ).to(device)

    if _is_autoencoder(encoder_type):
        aux_criterion = nn.MSELoss()
    else:
        aux_criterion = nn.CrossEntropyLoss()

    contrastive_loss = hp.get('contrastive_loss', 'supcon')
    criteria = build_contrastive_criterion(
        contrastive_loss,
        supcon_temperature=hp.get('supcon_temperature', 0.1),
        rnc_temperature=hp.get('rnc_temperature', 2.0),
        rnc_similarity=hp.get('rnc_similarity', 'cosine'),
        deepsad_margin=hp.get('deepsad_margin', 1.0),
        deepsad_repulsion_weight=hp.get('deepsad_repulsion_weight', 1.0),
    )
    lambda_supcon = hp.get('lambda_supcon', 0.5) if criteria.get('supcon') is not None else 0.0
    lambda_rnc = hp.get('lambda_rnc', 0.5) if criteria.get('rnc') is not None else 0.0
    lambda_deepsad = hp.get('lambda_deepsad', 1.0) if criteria.get('deepsad') is not None else 0.0

    warmup_epochs = int(hp.get('warmup_epochs', 0))
    optimizer = Adam(model.parameters(), lr=hp['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # Start at a low LR for warmup (epoch 0 gets lr/warmup_epochs)
    if warmup_epochs > 0:
        for pg in optimizer.param_groups:
            pg['lr'] = hp['lr'] / warmup_epochs

    history: Dict[str, List[float]] = {
        'train_aux': [], 'train_supcon': [], 'train_rnc': [], 'train_deepsad': [], 'train_total': [],
        'val_aux': [], 'val_supcon': [], 'val_rnc': [], 'val_deepsad': [], 'val_total': [],
        'lr': [],
        'aux_label': aux_label,
        'contrastive_loss': contrastive_loss,
    }

    # DeepSAD: initialize frozen center from all train-NoFlow projections.
    # Must happen after model build (weights initialized) and before epoch loop.
    if criteria.get('deepsad') is not None:
        if verbose:
            print("  Initialising DeepSAD center from train-NoFlow projections...")
        n_noflow = _init_deepsad_center(
            model, data['train_patches'], data['train_labels'],
            criteria['deepsad'], device,
        )
        if verbose:
            c_norm = criteria['deepsad'].center.norm().item()
            print(
                f"  → {n_noflow} NoFlow patches, center norm={c_norm:.4f}, "
                f"margin={hp.get('deepsad_margin', 1.0):.3f}, "
                f"repulsion_weight={hp.get('deepsad_repulsion_weight', 1.0):.3f}"
            )

    best_val = float('inf')
    epochs_no_improve = 0
    best_state = None
    best_epoch = 0

    if verbose:
        print(f"\n  Encoder type:       {encoder_type}")
        print(f"  Batches per epoch:  {len(train_loader)}")
        print(f"  Batch size:         {batch_size}  ({hp['k_per_series']} × series)")
        print(f"  Model parameters:   {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        tr_aux, tr_sc, tr_rn, tr_ds, tr_tot = _train_one_epoch(
            model, train_loader, optimizer, aux_criterion, criteria,
            lambda_supcon, lambda_rnc, lambda_deepsad, device, encoder_type,
        )
        history['train_aux'].append(tr_aux)
        history['train_supcon'].append(tr_sc)
        history['train_rnc'].append(tr_rn)
        history['train_deepsad'].append(tr_ds)
        history['train_total'].append(tr_tot)

        if val_loader is not None:
            va_aux, va_sc, va_rn, va_ds, va_tot = _eval_losses(
                model, val_loader, aux_criterion, criteria,
                lambda_supcon, lambda_rnc, lambda_deepsad, device, encoder_type,
            )
        else:
            va_aux, va_sc, va_rn, va_ds, va_tot = tr_aux, tr_sc, tr_rn, tr_ds, tr_tot

        history['val_aux'].append(va_aux)
        history['val_supcon'].append(va_sc)
        history['val_rnc'].append(va_rn)
        history['val_deepsad'].append(va_ds)
        history['val_total'].append(va_tot)
        # LR warmup: linear ramp for first warmup_epochs, then hand off to scheduler
        if epoch < warmup_epochs:
            warmup_lr = hp['lr'] * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step(va_tot)

        history['lr'].append(optimizer.param_groups[0]['lr'])

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"  ep {epoch+1:3d}/{epochs} | "
                f"tr {aux_label}={tr_aux:.4f} sc={tr_sc:.4f} rn={tr_rn:.4f} ds={tr_ds:.4f} tot={tr_tot:.4f} | "
                f"va {aux_label}={va_aux:.4f} sc={va_sc:.4f} rn={va_rn:.4f} ds={va_ds:.4f} tot={va_tot:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        # Only track best / early-stopping after warmup completes
        if epoch >= warmup_epochs:
            improved = va_tot < best_val
            if improved:
                best_val = va_tot
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

        if optuna_trial is not None:
            import optuna
            optuna_trial.report(va_tot, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch + 1} (best={best_epoch + 1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if center_out is not None and criteria.get('deepsad') is not None:
        center_out['deepsad_center'] = criteria['deepsad'].center.detach().cpu().clone()

    return model, history, best_epoch


def _evaluate_on_test(
    model: nn.Module,
    data: Dict,
    device: torch.device,
    gmm_n_components: int,
    gmm_covariance_type: str,
    test_files: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
    """
    For each unique test series: fit GMM on that series' NoFlow latents,
    score all latents, return ROC-AUC + adaptive-F1.
    Also returns an 'overall' entry with the mean.

    If test_files is provided (one file stem per patch, same length as
    test_patches), also returns per-file majority-vote results as the
    second element of the tuple. Otherwise the second element is {}.
    """
    if len(data['test_patches']) == 0:
        return {}, {}

    test_latents = _extract_latents(model, data['test_patches'], device)
    test_labels = np.asarray(data['test_labels'])
    test_series = np.array(data['test_series'])

    if test_files is not None and len(test_files) != len(test_labels):
        print(
            f"  [warn] test_files length {len(test_files)} != "
            f"test_labels {len(test_labels)} — skipping majority vote"
        )
        test_files = None

    out: Dict[str, Dict[str, float]] = {}
    mv_out: Dict[str, Dict] = {}
    aucs: List[float] = []

    for s in sorted(set(data['test_series'])):
        mask = test_series == s
        if mask.sum() == 0:
            continue
        lat_s = test_latents[mask]
        lbl_s = test_labels[mask]

        noflow_mask = lbl_s == 0
        if noflow_mask.sum() < max(gmm_n_components * 2, 4):
            continue

        noflow_lat = lat_s[noflow_mask]
        try:
            scores = _gmm_score(
                noflow_lat, lat_s,
                n_components=gmm_n_components,
                covariance_type=gmm_covariance_type,
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # Rank-deficient NoFlow covariance (latent collapse at high latent_dim
            # under strong SupCon pressure). Skip this series instead of killing
            # the Optuna trial / training run.
            print(f"  [gmm] skipping series {s}: {type(e).__name__}: {e}")
            continue

        auc = _auc_safe(lbl_s, scores)
        f1, thr = _f1_with_gauss_threshold(
            noflow_scores=scores[noflow_mask],
            all_labels=lbl_s,
            all_scores=scores,
        )
        # Accuracy, correct/wrong patch counts, confusion matrix — added
        # alongside the existing roc_auc/f1/threshold, none of which change.
        cls_metrics = _classification_metrics_at_threshold(lbl_s, scores, thr)
        out[s] = {
            'roc_auc': auc,
            'f1': f1,
            'threshold': thr,
            'n_noflow': int(noflow_mask.sum()),
            'n_flow': int((lbl_s == 1).sum()),
            **cls_metrics,
        }
        aucs.append(auc)

        if test_files is not None:
            files_s = test_files[mask]
            preds_s = (scores >= thr).astype(int)
            series_mv: Dict[str, Dict] = {}
            for fname in sorted(set(files_s)):
                fm = files_s == fname
                file_labels = lbl_s[fm]
                file_preds = preds_s[fm]
                n_nf = int((file_preds == 0).sum())
                n_lk = int((file_preds == 1).sum())
                majority_pred = 'LEAKAGE' if n_lk > n_nf else 'NOFLOW'
                true_lbl = 'LEAKAGE' if (file_labels == 1).any() else 'NOFLOW'
                series_mv[fname] = {
                    'true_label': true_lbl,
                    'majority_pred': majority_pred,
                    'n_noflow': n_nf,
                    'n_leakage': n_lk,
                }
            mv_out[s] = series_mv

    if aucs:
        out['overall'] = {
            'mean_roc_auc': float(np.mean(aucs)),
            'mean_f1': float(np.mean([v['f1'] for k, v in out.items() if k != 'overall'])),
            'n_series': len(aucs),
            **_aggregate_classification_metrics(out),
        }
    return out, mv_out


# ===========================================================================
# PART 2 — Fair per-site evaluation (chronological NoFlow calibration/test
# split). See study note:
# _mnt_d_projekte_KILD_inoson_kild_study_center_check_and_noflow_split.txt
#
# `_evaluate_on_test` above fits the per-site GMM on a series' NoFlow and
# then scores that SAME NoFlow again -> optimistic specificity/AUC (that
# NoFlow "looks normal" because the detector was fit on exactly it). The
# functions below fix that: split each series' NoFlow FILES chronologically
# into an early "calibration" set (fit + q95 threshold only, never scored)
# and a later, disjoint "test" set (scored, together with all Leak files).
# The old fit-and-score-on-same-NoFlow behaviour is preserved ONLY under the
# explicit mode='optimistic_upper_bound' — never the default.
# ===========================================================================

_DATE_RE = re.compile(r'(\d{4})_(\d{2})_(\d{2})')


def _extract_file_date(file_name: str) -> Optional[str]:
    """'hb_2026_06_12 1840_1759_no_leckage' -> '2026-06-12'. None if unparseable."""
    m = _DATE_RE.search(file_name)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{mo}-{d}"


def _q95_threshold(calibration_scores: np.ndarray) -> float:
    """95th percentile of calibration-NoFlow scores — the literal "q95 threshold"
    named in the study note (distinct from the mu+3*sigma Gaussian threshold
    `_f1_with_gauss_threshold` uses elsewhere in this module)."""
    return float(np.percentile(calibration_scores, 95))


def _mean_only_score(calibration_latents: np.ndarray, eval_latents: np.ndarray) -> np.ndarray:
    """Mode E fallback for small calibration sets: Euclidean distance to the
    calibration mean only, no covariance estimate — avoids fitting an
    unstable full Sigma from too few patches (study note: <500 patches)."""
    mu = calibration_latents.mean(axis=0)
    return np.linalg.norm(eval_latents - mu, axis=1)


def _split_calibration_test_files(
    series_arr: np.ndarray,
    files_arr: np.ndarray,
    labels_arr: np.ndarray,
    calibration_file_ratio: float = 0.5,
) -> Dict[str, Dict]:
    """
    Per series: group patches by file, classify each file NoFlow-only vs Leak
    (any Flow-labelled patch in the file -> Leak, same convention majority
    voting already uses), sort NoFlow files chronologically by date parsed
    from the filename, and split the earliest
    ``ceil(n_noflow_files * calibration_file_ratio)`` (min 1, leaving >=1 for
    test when possible) into calibration — the rest into test. All Leak files
    always go to test, never calibration.
    """
    split: Dict[str, Dict] = {}
    for s in sorted(set(series_arr.tolist())):
        smask = series_arr == s
        files_s = files_arr[smask]
        labels_s = labels_arr[smask]

        file_is_leak: Dict[str, bool] = {}
        file_dates: Dict[str, Optional[str]] = {}
        for fname in sorted(set(files_s.tolist())):
            fmask = files_s == fname
            file_is_leak[fname] = bool((labels_s[fmask] == 1).any())
            file_dates[fname] = _extract_file_date(fname)

        noflow_files = sorted(
            [f for f, is_leak in file_is_leak.items() if not is_leak],
            key=lambda f: (file_dates[f] is None, file_dates[f] or '', f),
        )
        leak_files = [f for f, is_leak in file_is_leak.items() if is_leak]

        if len(noflow_files) <= 1:
            # Can't split a single NoFlow file into disjoint calibration+test.
            # Everything goes to calibration; test side has no NoFlow (still
            # meaningful for AUC ranking against Leak, not for specificity —
            # flagged downstream via n_test_noflow_patches == 0).
            calibration_files = list(noflow_files)
            test_noflow_files: List[str] = []
        else:
            n_calib = max(1, math.ceil(len(noflow_files) * calibration_file_ratio))
            n_calib = min(n_calib, len(noflow_files) - 1)  # always leave >=1 file for test
            calibration_files = noflow_files[:n_calib]
            test_noflow_files = noflow_files[n_calib:]

        calib_dates = sorted(d for f in calibration_files if (d := file_dates[f]) is not None)
        # Leak files can fall chronologically BETWEEN NoFlow days in real data
        # (e.g. leak on the 19th/21st/23rd, NoFlow on the 18th/20th/22nd/24th/
        # 25th) — a leak is real regardless of when it happens relative to
        # calibration, it just must never BE the calibration data. So the
        # "calibration precedes test" ordering check (study note: "calibration
        # NoFlow must lie in time BEFORE the test NoFlow") only makes sense
        # against test NoFlow specifically, not against interleaved Leak
        # dates — tracked separately here.
        test_noflow_dates = sorted(d for f in test_noflow_files if (d := file_dates[f]) is not None)
        test_dates = sorted(
            d for f in (test_noflow_files + leak_files) if (d := file_dates[f]) is not None
        )

        split[s] = {
            'calibration_files': calibration_files,
            'test_noflow_files': test_noflow_files,
            'leak_files': leak_files,
            'calibration_date_range': [calib_dates[0], calib_dates[-1]] if calib_dates else [None, None],
            'test_date_range': [test_dates[0], test_dates[-1]] if test_dates else [None, None],
            'test_noflow_date_range': (
                [test_noflow_dates[0], test_noflow_dates[-1]] if test_noflow_dates else [None, None]
            ),
        }
    return split


def _evaluate_on_test_fair(
    model: nn.Module,
    data: Dict,
    device: torch.device,
    gmm_n_components: int,
    gmm_covariance_type: str,
    test_files: Optional[np.ndarray] = None,
    mode: str = 'fair',
    calibration_file_ratio: float = 0.5,
    min_calibration_patches: int = 500,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Corrected per-site evaluation: calibration NoFlow (earliest files, per
    series) fits the GMM + q95 threshold; test NoFlow (later, disjoint files)
    + all Leak files are scored. Calibration NoFlow never appears in any
    reported AUC/F1/specificity/confusion-matrix/majority-vote value.

    mode='fair' (default): the corrected calibration/test split above.
    mode='optimistic_upper_bound': delegates to the ORIGINAL `_evaluate_on_test`
        (fits AND scores on the same series NoFlow) — preserved only for
        explicit side-by-side comparison. Must be requested by name, never
        the default (see study note "IMPORTANT" section).

    Returns (metrics_dict, majority_vote_dict, fold_records_dict). Every
    fold_records entry carries calibration/test file lists, patch counts per
    role, date ranges, gmm_mode, threshold, roc_auc, f1, specificity, and a
    per-file majority-vote correctness map — the full audit trail the study
    note's step 13 asks for. fold_records is {} under optimistic_upper_bound
    (that path predates this bookkeeping and isn't the protocol being audited).
    """
    if mode == 'optimistic_upper_bound':
        metrics, mv = _evaluate_on_test(
            model, data, device, gmm_n_components, gmm_covariance_type, test_files,
        )
        for k, v in metrics.items():
            if k != 'overall':
                v['gmm_mode'] = 'optimistic_upper_bound'
        return metrics, mv, {}

    if mode != 'fair':
        raise ValueError(f"Unknown mode {mode!r} — must be 'fair' or 'optimistic_upper_bound'.")

    if len(data['test_patches']) == 0 or test_files is None:
        return {}, {}, {}

    test_latents = _extract_latents(model, data['test_patches'], device)
    test_labels = np.asarray(data['test_labels'])
    test_series = np.array(data['test_series'])

    if len(test_files) != len(test_labels):
        print(f"  [warn] test_files length {len(test_files)} != test_labels {len(test_labels)} "
              f"— cannot do file-level fair split, skipping.")
        return {}, {}, {}

    splits = _split_calibration_test_files(test_series, test_files, test_labels, calibration_file_ratio)

    out: Dict[str, Dict] = {}
    mv_out: Dict[str, Dict] = {}
    fold_records: Dict[str, Dict] = {}
    aucs: List[float] = []

    for s, sp in splits.items():
        smask = test_series == s
        lat_s = test_latents[smask]
        lbl_s = test_labels[smask]
        files_s = test_files[smask]

        calib_file_set = set(sp['calibration_files'])
        test_file_set = set(sp['test_noflow_files']) | set(sp['leak_files'])

        # Assertion 1: calibration and test file IDs are disjoint.
        assert not (calib_file_set & test_file_set), (
            f"[{s}] calibration/test file sets overlap: {calib_file_set & test_file_set}"
        )

        calib_mask = np.isin(files_s, list(calib_file_set)) if calib_file_set else np.zeros(len(files_s), dtype=bool)
        scored_mask = np.isin(files_s, list(test_file_set)) if test_file_set else np.zeros(len(files_s), dtype=bool)

        # Assertion 2: calibration and scored patches are disjoint.
        assert not np.any(calib_mask & scored_mask), f"[{s}] calibration/scored patches overlap."

        # Assertion 3: calibration dates precede test NoFlow dates (when known).
        # Compared against test NoFlow specifically, NOT combined test+Leak —
        # Leak files can fall chronologically between NoFlow days in real data
        # (e.g. hb-25.05.2026: leak on 19th/21st/23rd, NoFlow on 18th/20th/
        # 22nd/24th/25th) and are real regardless of when they happen, they
        # just must never be part of calibration (see _split_calibration_test_files).
        c_lo, c_hi = sp['calibration_date_range']
        tnf_lo, tnf_hi = sp['test_noflow_date_range']
        if c_hi is not None and tnf_lo is not None:
            assert c_hi <= tnf_lo, (
                f"[{s}] calibration date range {sp['calibration_date_range']} does not precede "
                f"test NoFlow date range {sp['test_noflow_date_range']}."
            )

        n_calib = int(calib_mask.sum())
        n_scored = int(scored_mask.sum())
        n_test_noflow = int((scored_mask & (lbl_s == 0)).sum())
        n_leak = int((scored_mask & (lbl_s == 1)).sum())

        record: Dict = {
            'calibration_files': sp['calibration_files'],
            'test_noflow_files': sp['test_noflow_files'],
            'leak_files': sp['leak_files'],
            'n_calibration_patches': n_calib,
            'n_test_noflow_patches': n_test_noflow,
            'n_leak_patches': n_leak,
            'calibration_date_range': sp['calibration_date_range'],
            'test_date_range': sp['test_date_range'],
            'test_noflow_date_range': sp['test_noflow_date_range'],
        }

        if n_calib == 0 or n_scored == 0 or n_test_noflow == 0:
            record['skipped_reason'] = (
                'no calibration NoFlow' if n_calib == 0 else
                'no scored patches' if n_scored == 0 else
                'no disjoint test NoFlow (series has <=1 NoFlow file, see _split_calibration_test_files)'
            )
            fold_records[s] = record
            continue

        calib_latents = lat_s[calib_mask]
        scored_latents = lat_s[scored_mask]
        scored_labels = lbl_s[scored_mask]

        if n_calib < min_calibration_patches:
            # Point 14: flag + fall back to mean-only, do not silently fit an
            # unstable full covariance from too little data.
            gmm_mode = 'mean_only_fallback'
            record['flagged_small_calibration'] = True
            calib_scores = _mean_only_score(calib_latents, calib_latents)
            scores = _mean_only_score(calib_latents, scored_latents)
        else:
            gmm_mode = 'gmm_full_covariance'
            record['flagged_small_calibration'] = False
            if n_calib < max(gmm_n_components * 2, 4):
                record['skipped_reason'] = 'calibration too small for GMM n_components'
                fold_records[s] = record
                continue
            try:
                gmm = GaussianMixture(
                    n_components=gmm_n_components, covariance_type=gmm_covariance_type,
                    reg_covar=1e-3, random_state=42,
                )
                gmm.fit(calib_latents)
                calib_scores = -gmm.score_samples(calib_latents)
                scores = -gmm.score_samples(scored_latents)
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"  [gmm] skipping series {s}: {type(e).__name__}: {e}")
                record['skipped_reason'] = f'GMM fit failed: {type(e).__name__}: {e}'
                fold_records[s] = record
                continue

        threshold = _q95_threshold(calib_scores)
        auc = _auc_safe(scored_labels, scores)
        cls_metrics = _classification_metrics_at_threshold(scored_labels, scores, threshold)
        cm = cls_metrics['confusion_matrix']
        specificity = cm['tn'] / max(cm['tn'] + cm['fp'], 1)
        precision = cm['tp'] / max(cm['tp'] + cm['fp'], 1)
        recall = cm['tp'] / max(cm['tp'] + cm['fn'], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        preds = (scores > threshold).astype(int)

        out[s] = {
            'roc_auc': auc,
            'f1': f1,
            'specificity': specificity,
            'threshold': threshold,
            'gmm_mode': gmm_mode,
            'n_noflow': n_test_noflow,
            'n_flow': n_leak,
            **cls_metrics,
        }
        aucs.append(auc)

        # Majority vote — test_noflow_files + leak_files ONLY, never calibration.
        series_mv: Dict[str, Dict] = {}
        for fname in sorted(test_file_set):
            fm = files_s[scored_mask] == fname
            if fm.sum() == 0:
                continue
            file_labels = scored_labels[fm]
            file_preds = preds[fm]
            n_nf = int((file_preds == 0).sum())
            n_lk = int((file_preds == 1).sum())
            majority_pred = 'LEAKAGE' if n_lk > n_nf else 'NOFLOW'
            true_lbl = 'LEAKAGE' if (file_labels == 1).any() else 'NOFLOW'
            series_mv[fname] = {
                'true_label': true_lbl, 'majority_pred': majority_pred,
                'n_noflow': n_nf, 'n_leakage': n_lk,
            }

        # Assertion 4: no calibration file ever appears in the scored
        # predictions / majority-vote output (redundant with assertions 1-2
        # given how series_mv is built, kept explicit per the study note).
        assert not (set(series_mv.keys()) & calib_file_set), (
            f"[{s}] calibration file leaked into majority-vote predictions: "
            f"{set(series_mv.keys()) & calib_file_set}"
        )
        mv_out[s] = series_mv

        record.update({
            'gmm_mode': gmm_mode, 'threshold': threshold,
            'roc_auc': auc, 'f1': f1, 'specificity': specificity,
            'majority_vote_correct': {
                fn: (v['true_label'] == v['majority_pred']) for fn, v in series_mv.items()
            },
        })
        fold_records[s] = record

    if aucs:
        out['overall'] = {
            'mean_roc_auc': float(np.mean(aucs)),
            'mean_f1': float(np.mean([v['f1'] for k, v in out.items() if k != 'overall'])),
            'mean_specificity': float(np.mean([v['specificity'] for k, v in out.items() if k != 'overall'])),
            'n_series': len(aucs),
            **_aggregate_classification_metrics(out),
        }
    return out, mv_out, fold_records


def _evaluate_global_gmm(
    model: nn.Module,
    data: Dict,
    device: torch.device,
    gmm_n_components: int,
    gmm_covariance_type: str,
    test_files: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
    """
    Single global GMM fitted on encoder-train NoFlow latents, scored against
    all test patches. Per-series ROC-AUC reported alongside an overall mean.

    This is the deployment-realistic metric: the GMM is calibrated once on
    whatever NoFlow data was seen during the encoder's training period, and
    must remain valid for every series in the eval set without per-site
    refit. Diverges from :func:`_evaluate_on_test`, which fits a fresh GMM
    per eval series on that series' own NoFlow.

    If `test_files` is provided (one file stem per patch, same length as
    test_patches), also returns per-file majority-vote results as the second
    element of the tuple, mirroring `_evaluate_on_test`. Otherwise the
    second element is {}.
    """
    if len(data['test_patches']) == 0 or len(data['train_patches']) == 0:
        return {}, {}

    train_labels = np.asarray(data['train_labels'])
    train_noflow_mask = train_labels == 0
    if train_noflow_mask.sum() < max(gmm_n_components * 2, 4):
        print(
            f"  [global gmm] not enough train NoFlow patches "
            f"({int(train_noflow_mask.sum())}); skipping."
        )
        return {}, {}

    train_latents_all = _extract_latents(model, data['train_patches'], device)
    # Only NoFlow latents from the TRAINING pool feed the fit below — the
    # held-out/eval series never contributes data to this GMM.
    train_noflow_lat = train_latents_all[train_noflow_mask]

    # ═══════════════════════════════════════════════════════════════════
    # GLOBAL GMM FIT — the single fit for this whole evaluation.
    # Fitted ONCE on the training pool's NoFlow latents only (never on any
    # test/held-out series' own data). Every eval series below is scored
    # against this one model with no per-site refit — this is what makes
    # the metric "global" rather than per-series/site-recalibrated.
    # ═══════════════════════════════════════════════════════════════════
    try:
        gmm = GaussianMixture(
            n_components=gmm_n_components,
            covariance_type=gmm_covariance_type,
            reg_covar=1e-3,
            random_state=42,
        )
        gmm.fit(train_noflow_lat)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"  [global gmm] fit failed: {type(e).__name__}: {e}")
        return {}, {}
    # ═══════════════════════════════════════════════════════════════════

    test_latents = _extract_latents(model, data['test_patches'], device)
    test_labels = np.asarray(data['test_labels'])
    test_series = np.array(data['test_series'])
    # Score every eval patch against the single global GMM fit above —
    # no refitting per series happens anywhere below this line.
    scores_all = -gmm.score_samples(test_latents)

    if test_files is not None and len(test_files) != len(test_labels):
        print(
            f"  [warn] test_files length {len(test_files)} != "
            f"test_labels {len(test_labels)} — skipping majority vote"
        )
        test_files = None

    out: Dict[str, Dict[str, float]] = {}
    mv_out: Dict[str, Dict] = {}
    aucs: List[float] = []

    for s in sorted(set(data['test_series'])):
        mask = test_series == s
        if mask.sum() == 0:
            continue
        lbl_s = test_labels[mask]
        scores_s = scores_all[mask]
        noflow_mask_s = lbl_s == 0
        if noflow_mask_s.sum() == 0 or (lbl_s == 1).sum() == 0:
            continue

        auc = _auc_safe(lbl_s, scores_s)
        # Threshold from this series' own NoFlow scores under the GLOBAL GMM
        # — this is *not* a per-series GMM, the threshold just adapts to each
        # site's local score distribution under the one global model.
        f1, thr = _f1_with_gauss_threshold(
            noflow_scores=scores_s[noflow_mask_s],
            all_labels=lbl_s,
            all_scores=scores_s,
        )
        # Accuracy, correct/wrong patch counts, confusion matrix — added
        # alongside the existing roc_auc/f1/threshold, none of which change.
        cls_metrics = _classification_metrics_at_threshold(lbl_s, scores_s, thr)
        out[s] = {
            'roc_auc': auc,
            'f1': f1,
            'threshold': thr,
            'n_noflow': int(noflow_mask_s.sum()),
            'n_flow': int((lbl_s == 1).sum()),
            **cls_metrics,
        }
        aucs.append(auc)

        if test_files is not None:
            files_s = test_files[mask]
            preds_s = (scores_s >= thr).astype(int)
            series_mv: Dict[str, Dict] = {}
            for fname in sorted(set(files_s)):
                fm = files_s == fname
                file_labels = lbl_s[fm]
                file_preds = preds_s[fm]
                n_nf = int((file_preds == 0).sum())
                n_lk = int((file_preds == 1).sum())
                majority_pred = 'LEAKAGE' if n_lk > n_nf else 'NOFLOW'
                true_lbl = 'LEAKAGE' if (file_labels == 1).any() else 'NOFLOW'
                series_mv[fname] = {
                    'true_label': true_lbl,
                    'majority_pred': majority_pred,
                    'n_noflow': n_nf,
                    'n_leakage': n_lk,
                }
            mv_out[s] = series_mv

    if aucs:
        out['overall'] = {
            'mean_roc_auc': float(np.mean(aucs)),
            'mean_f1': float(np.mean([v['f1'] for k, v in out.items() if k != 'overall'])),
            'n_series': len(aucs),
            'gmm_fit_n_noflow': int(train_noflow_mask.sum()),
            **_aggregate_classification_metrics(out),
        }
    return out, mv_out


def _evaluate_global_gmm_conditional(
    model: nn.Module,
    data: Dict,
    device: torch.device,
    gmm_n_components: int,
    gmm_covariance_type: str,
    temp_maps: Dict[str, Dict[str, float]],
    temp_target_names: List[str] = ('temp_pipe',),
    test_files: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
    """
    Temperature-conditioned global GMM. Linear regression
    ``z_i = A_i @ slope + intercept`` is fitted per latent dim on encoder-train
    NoFlow latents against one OR MORE per-series mean temperature covariates
    (``temp_target_names``, e.g. just ``['temp_pipe']`` or jointly
    ``['temp_pipe', 'temp_amb']``). The GMM is then fitted on the residuals
    (z − z_pred(T)). At eval time, each series' temperature(s) are plugged
    into the regression to produce that series' predicted NoFlow centroid;
    eval-patch latents are scored on their residuals against the residual GMM.

    `temp_maps` is keyed by target name: ``{'temp_pipe': {series: value}, ...}``
    — one entry per name in `temp_target_names`. With a single target this is
    the original single-covariate regression (unchanged behaviour/output);
    with two, it's a joint two-covariate regression (e.g. pipe + ambient).

    If `test_files` is provided (one file stem per patch, same length as
    test_patches), also returns per-file majority-vote results as the second
    element of the tuple, mirroring `_evaluate_global_gmm`/`_evaluate_on_test`.
    Otherwise the second element is {}.

    This embeds the centroid-trajectory drift model from CLAUDE.md directly
    into scoring — no separate post-processing step. Failure mode: if the
    drift is not approximately linear in T (e.g. the April-cliff regime where
    centroid jumps 9 orders of magnitude), the regression cannot capture it
    and the eval patches at OOD T will appear as residual-space outliers.
    """
    temp_target_names = list(temp_target_names)
    n_targets = len(temp_target_names)

    if len(data['test_patches']) == 0 or len(data['train_patches']) == 0:
        return {}, {}
    if not temp_maps or any(not temp_maps.get(t) for t in temp_target_names):
        print(f"  [conditional gmm] empty temp_map for one of {temp_target_names}; skipping.")
        return {}, {}

    train_labels = np.asarray(data['train_labels'])
    train_series = np.array(data['train_series'])
    train_noflow_mask = train_labels == 0
    if train_noflow_mask.sum() < max(gmm_n_components * 2, 4):
        print(
            f"  [conditional gmm] not enough train NoFlow patches "
            f"({int(train_noflow_mask.sum())}); skipping."
        )
        return {}, {}

    # Per-patch temperatures from the per-series mapping — one column per
    # target, "if available" handled by NaN-dropping any patch missing ANY
    # of the requested targets (row-wise AND across columns).
    train_temps = np.stack(
        [
            np.array([temp_maps[t].get(s, np.nan) for s in train_series], dtype=np.float64)
            for t in temp_target_names
        ],
        axis=1,
    )  # shape (n_patches, n_targets)
    valid_train = ~np.isnan(train_temps).any(axis=1)
    if not valid_train.all():
        missing = sorted(set(np.array(train_series)[~valid_train].tolist()))
        print(
            f"  [conditional gmm] {temp_target_names} missing for series "
            f"{missing} — those patches dropped from regression fit."
        )
    fit_mask = train_noflow_mask & valid_train

    train_latents_all = _extract_latents(model, data['train_patches'], device)
    z_fit = train_latents_all[fit_mask]
    T_fit = train_temps[fit_mask]  # shape (n_fit, n_targets)

    # "if available" — each covariate individually needs ≥2 distinct values,
    # otherwise the regression on that axis is undefined; skip the whole
    # setting rather than silently dropping a column.
    n_unique_per_target = [len(np.unique(T_fit[:, j])) for j in range(n_targets)]
    if any(n < 2 for n in n_unique_per_target):
        print(
            f"  [conditional gmm] insufficient distinct training values "
            f"{dict(zip(temp_target_names, n_unique_per_target))}; "
            f"regression undefined, skipping."
        )
        return {}, {}

    # Per-latent-dim linear regression z_i = T @ slope_i + intercept_i.
    # coef shape (n_targets + 1, latent_dim) — last row = intercept.
    A = np.hstack([T_fit, np.ones((T_fit.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, z_fit, rcond=None)
    slope = coef[:n_targets]       # shape (n_targets, latent_dim)
    intercept = coef[n_targets]    # shape (latent_dim,)

    z_pred_train = T_fit @ slope + intercept[None, :]
    residual_train = z_fit - z_pred_train

    try:
        gmm = GaussianMixture(
            n_components=gmm_n_components,
            covariance_type=gmm_covariance_type,
            reg_covar=1e-3,
            random_state=42,
        )
        gmm.fit(residual_train)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"  [conditional gmm] fit failed: {type(e).__name__}: {e}")
        return {}, {}

    # Eval side
    test_latents = _extract_latents(model, data['test_patches'], device)
    test_labels = np.asarray(data['test_labels'])
    test_series = np.array(data['test_series'])

    if test_files is not None and len(test_files) != len(test_labels):
        print(
            f"  [warn] test_files length {len(test_files)} != "
            f"test_labels {len(test_labels)} — skipping majority vote"
        )
        test_files = None

    out: Dict[str, Dict[str, float]] = {}
    mv_out: Dict[str, Dict] = {}
    aucs: List[float] = []

    for s in sorted(set(data['test_series'])):
        mask = test_series == s
        if mask.sum() == 0:
            continue
        T_s_list = [temp_maps[t].get(s, np.nan) for t in temp_target_names]
        if any(np.isnan(v) for v in T_s_list):
            print(
                f"  [conditional gmm] no {temp_target_names} for {s}; skipping."
            )
            continue
        lbl_s = test_labels[mask]
        noflow_mask_s = lbl_s == 0
        if noflow_mask_s.sum() == 0 or (lbl_s == 1).sum() == 0:
            continue

        T_s = np.array(T_s_list, dtype=np.float64)  # shape (n_targets,)
        z_pred_s = T_s @ slope + intercept  # shape (latent_dim,)
        residual_s = test_latents[mask] - z_pred_s[None, :]
        scores_s = -gmm.score_samples(residual_s)

        auc = _auc_safe(lbl_s, scores_s)
        f1, thr = _f1_with_gauss_threshold(
            noflow_scores=scores_s[noflow_mask_s],
            all_labels=lbl_s,
            all_scores=scores_s,
        )
        # Accuracy, correct/wrong patch counts, confusion matrix — added
        # alongside the existing roc_auc/f1/threshold, none of which change.
        cls_metrics = _classification_metrics_at_threshold(lbl_s, scores_s, thr)
        out[s] = {
            'roc_auc': auc,
            'f1': f1,
            'threshold': thr,
            **{t: float(v) for t, v in zip(temp_target_names, T_s_list)},
            'n_noflow': int(noflow_mask_s.sum()),
            'n_flow': int((lbl_s == 1).sum()),
            **cls_metrics,
        }
        aucs.append(auc)

        if test_files is not None:
            files_s = test_files[mask]
            preds_s = (scores_s >= thr).astype(int)
            series_mv: Dict[str, Dict] = {}
            for fname in sorted(set(files_s)):
                fm = files_s == fname
                file_labels = lbl_s[fm]
                file_preds = preds_s[fm]
                n_nf = int((file_preds == 0).sum())
                n_lk = int((file_preds == 1).sum())
                majority_pred = 'LEAKAGE' if n_lk > n_nf else 'NOFLOW'
                true_lbl = 'LEAKAGE' if (file_labels == 1).any() else 'NOFLOW'
                series_mv[fname] = {
                    'true_label': true_lbl,
                    'majority_pred': majority_pred,
                    'n_noflow': n_nf,
                    'n_leakage': n_lk,
                }
            mv_out[s] = series_mv

    if aucs:
        out['overall'] = {
            'mean_roc_auc': float(np.mean(aucs)),
            'mean_f1': float(np.mean([v['f1'] for k, v in out.items() if k != 'overall'])),
            'n_series': len(aucs),
            **_aggregate_classification_metrics(out),
            'gmm_fit_n_noflow': int(fit_mask.sum()),
            'temp_targets': temp_target_names,
            'n_train_temperatures': dict(zip(temp_target_names, n_unique_per_target)),
            'slope_norm': float(np.linalg.norm(slope)),
        }
    return out, mv_out


def _evaluate_global_gmm_hierarchical(
    model: nn.Module,
    data: Dict,
    device: torch.device,
    gmm_n_components: int,
    gmm_covariance_type: str,
    temp_map: Dict[str, float],
    temp_target_name: str = 'temp_pipe',
) -> Dict[str, Dict[str, float]]:
    """
    Hierarchical conditional GMM that separates the centroid model from
    the within-series scatter model — addresses the core defect of
    :func:`_evaluate_global_gmm_conditional`, where the residual covariance
    after a global linear T-fit still carries leftover between-series
    variance and is therefore wider than any single series' actual NoFlow
    cluster.

    Recipe:

    1. Per-series NoFlow centroid μ_s = mean(z_train_NoFlow_s) for each
       training series s.
    2. Linear regression of these centroids against per-series temperature:
       μ_s ≈ a · T_s + b. (Fewer points = 7, but exactly the right
       quantity to fit — the centroid-vs-T trajectory.)
    3. Per-series-CENTERED residuals: r_i = z_i − μ_s(i) (subtract that
       series' actual centroid, not the regression-predicted one).
    4. Fit GMM on these residuals → Σ_within_pooled. This contains only
       intra-series scatter, with the between-series drift fully removed.
    5. Eval on series s: predict centroid via T-regression
       μ_pred = a · T_s + b, score (z − μ_pred) against the pooled
       Σ_within GMM.

    Why this can succeed where the simpler conditional GMM cannot: AUC is
    a within-series ranking metric, dominated by Σ_within. The simpler
    conditional GMM uses Σ_residual ≈ Σ_within + leftover_Σ_between
    (because linear T-fit doesn't fully explain the centroid drift), so
    its scoring "credibility region" is too wide. Hierarchical fitting
    pins Σ to exactly Σ_within_pooled, recovering the per-site GMM's
    discriminative power without per-site refit.

    Targets the Eval-B 18.04 gap (per-site 0.95 → global 0.88). If this
    closes the gap, the operative recipe at deployment becomes "ein
    Ein-Mal-Fit der Centroid-Trajektorie, danach Centroid-Extrapolation
    pro neuem Standort".
    """
    if len(data['test_patches']) == 0 or len(data['train_patches']) == 0:
        return {}
    if not temp_map:
        print("  [hierarchical gmm] empty temp_map; skipping.")
        return {}

    train_labels = np.asarray(data['train_labels'])
    train_series = np.array(data['train_series'])
    train_noflow_mask = train_labels == 0
    if train_noflow_mask.sum() < max(gmm_n_components * 2, 4):
        print(
            f"  [hierarchical gmm] not enough train NoFlow patches "
            f"({int(train_noflow_mask.sum())}); skipping."
        )
        return {}

    train_temps = np.array(
        [temp_map.get(s, np.nan) for s in train_series], dtype=np.float64
    )
    fit_mask = train_noflow_mask & ~np.isnan(train_temps)
    if fit_mask.sum() < 4:
        print(
            f"  [hierarchical gmm] not enough valid (NoFlow ∩ temp) "
            f"patches ({int(fit_mask.sum())}); skipping."
        )
        return {}

    train_latents_all = _extract_latents(model, data['train_patches'], device)
    z_fit = train_latents_all[fit_mask]
    series_fit = train_series[fit_mask]
    T_fit = train_temps[fit_mask]

    # Step 1: per-series NoFlow centroids (one point per series).
    unique_series = sorted(set(series_fit.tolist()))
    series_centroids = {}
    series_temps = {}
    for s in unique_series:
        mask = series_fit == s
        series_centroids[s] = z_fit[mask].mean(axis=0)
        series_temps[s] = float(T_fit[mask].mean())

    if len(unique_series) < 2:
        print(
            f"  [hierarchical gmm] only {len(unique_series)} training "
            f"series with valid temperature; regression undefined, skipping."
        )
        return {}

    # Step 2: regress centroids on per-series mean temperature.
    T_arr = np.array(
        [series_temps[s] for s in unique_series]
    ).reshape(-1, 1)
    C_arr = np.stack(
        [series_centroids[s] for s in unique_series], axis=0
    )
    A = np.hstack([T_arr, np.ones_like(T_arr)])
    coef, *_ = np.linalg.lstsq(A, C_arr, rcond=None)
    slope = coef[0]
    intercept = coef[1]

    # Step 3: per-series CENTERED residuals (each patch minus its own
    # series' centroid). This isolates within-series scatter from any
    # between-series component, including the part the linear fit
    # cannot explain.
    centroid_per_patch = np.stack(
        [series_centroids[s] for s in series_fit], axis=0
    )
    residuals_within = z_fit - centroid_per_patch

    # Step 4: GMM on within-series residuals → Σ_within_pooled.
    try:
        gmm = GaussianMixture(
            n_components=gmm_n_components,
            covariance_type=gmm_covariance_type,
            reg_covar=1e-3,
            random_state=42,
        )
        gmm.fit(residuals_within)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"  [hierarchical gmm] fit failed: {type(e).__name__}: {e}")
        return {}

    # Eval side: predicted centroid from T-regression, residual scored
    # against Σ_within GMM.
    test_latents = _extract_latents(model, data['test_patches'], device)
    test_labels = np.asarray(data['test_labels'])
    test_series_arr = np.array(data['test_series'])

    out: Dict[str, Dict[str, float]] = {}
    aucs: List[float] = []

    for s in sorted(set(data['test_series'])):
        mask = test_series_arr == s
        if mask.sum() == 0:
            continue
        if s not in temp_map or np.isnan(temp_map[s]):
            print(f"  [hierarchical gmm] no temperature for {s}; skipping.")
            continue
        lbl_s = test_labels[mask]
        noflow_mask_s = lbl_s == 0
        if noflow_mask_s.sum() == 0 or (lbl_s == 1).sum() == 0:
            continue

        T_s = float(temp_map[s])
        mu_pred = T_s * slope + intercept  # extrapolated centroid
        residuals_s = test_latents[mask] - mu_pred[None, :]
        scores_s = -gmm.score_samples(residuals_s)

        auc = _auc_safe(lbl_s, scores_s)
        f1, thr = _f1_with_gauss_threshold(
            noflow_scores=scores_s[noflow_mask_s],
            all_labels=lbl_s,
            all_scores=scores_s,
        )
        # Distance from training-T envelope — flags far-extrapolation
        # cells where the linear centroid model becomes untrustworthy.
        T_train_min = min(series_temps.values())
        T_train_max = max(series_temps.values())
        if T_s < T_train_min:
            t_extrap = T_train_min - T_s
        elif T_s > T_train_max:
            t_extrap = T_s - T_train_max
        else:
            t_extrap = 0.0
        # Accuracy, correct/wrong patch counts, confusion matrix — added
        # alongside the existing roc_auc/f1/threshold, none of which change.
        cls_metrics = _classification_metrics_at_threshold(lbl_s, scores_s, thr)
        out[s] = {
            'roc_auc': auc,
            'f1': f1,
            'threshold': thr,
            'temp': T_s,
            'temp_extrap_degC': t_extrap,
            'n_noflow': int(noflow_mask_s.sum()),
            'n_flow': int((lbl_s == 1).sum()),
            **cls_metrics,
        }
        aucs.append(auc)

    if aucs:
        # Σ_within trace as a single scalar diagnostic — useful for
        # comparing hierarchical vs. plain conditional fits.
        if gmm.covariance_type == 'full':
            sigma_trace = float(np.trace(gmm.covariances_[0]))
        else:
            sigma_trace = float('nan')
        out['overall'] = {
            'mean_roc_auc': float(np.mean(aucs)),
            'mean_f1': float(np.mean([v['f1'] for k, v in out.items() if k != 'overall'])),
            **_aggregate_classification_metrics(out),
            'n_series': len(aucs),
            'gmm_fit_n_residuals': int(fit_mask.sum()),
            'n_train_series': len(unique_series),
            'temp_target': temp_target_name,
            'temp_train_min': float(min(series_temps.values())),
            'temp_train_max': float(max(series_temps.values())),
            'slope_norm': float(np.linalg.norm(slope)),
            'sigma_within_trace': sigma_trace,
        }
    return out


# ===========================================================================
# Public entry points
# ===========================================================================

def train_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """Single training run with full diagnostics + test evaluation."""
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    img_out = (img_path / 'contrastive' / 'train') if img_path else None
    if img_out:
        img_out.mkdir(parents=True, exist_ok=True)

    # Load tuned hyperparameters first so the variant block can sync cfg
    # BEFORE encoder_type / loss-recipe fields are read downstream.
    best_params_path = results_dir / 'optuna_best_params.json'
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best = json.load(f)
        tuned = best.get('params', {})
        print(f"Using tuned hyperparameters from {best_params_path}")
        _apply_variant_from_best_params(cfg, best)
    else:
        tuned = {}
        print("No Optuna best params found — using config defaults.")

    encoder_type = cfg.get('encoder_type', 'autoencoder')
    encoder_arch = cfg.get('encoder_arch', 'cnn')
    print("=" * 60)
    print(
        f"CONTRASTIVE {encoder_arch.upper()} ENCODER ({encoder_type.upper()}) "
        f"— SINGLE TRAINING RUN"
    )
    print("=" * 60)
    print(f"Device: {device}")

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    # Resolve which series feed the encoder vs. the GMM eval. Defaults to
    # splits.json's train_series/test_series if no override is configured.
    encoder_train_series = cfg.get('encoder_train_series') or None
    gmm_eval_series = cfg.get('gmm_eval_series') or None
    train_ids, val_ids, test_ids, train_resolved, eval_resolved = (
        _resolve_train_eval_ids(
            splits_data, metadata_by_id,
            encoder_train_series, gmm_eval_series,
        )
    )
    print(f"  encoder_train_series: {train_resolved}")
    print(f"  gmm_eval_series:      {eval_resolved}")

    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)

    selected = _resolve_selected_features(cfg)

    data = _prepare_dataset_tensors(
        train_ids, val_ids, test_ids, prepared_base, metadata_by_id,
        selected,
        cfg.get('scaler', 'standard'),
        dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
        regress_target_map=regress_map,
    )

    _is_transformer = (encoder_arch == 'transformer')
    hp = {
        'encoder_type': encoder_type,
        'encoder_arch': encoder_arch,
        # CNN-specific keys: set to None when Transformer is active (back-compat)
        'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
        'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
        'dropout': cfg.get('dropout', 0.3),
        'latent_dim': tuned.get('latent_dim', cfg.get('latent_dim', 32)),
        'projection_dim': tuned.get('projection_dim', cfg.get('projection_dim', 64)),
        'contrastive_loss': cfg.get('contrastive_loss', 'supcon'),
        'lambda_supcon': tuned.get('lambda_supcon', cfg.get('lambda_supcon', 0.5)),
        'supcon_temperature': tuned.get('supcon_temperature', cfg.get('supcon_temperature', 0.1)),
        'lambda_rnc': tuned.get('lambda_rnc', cfg.get('lambda_rnc', 0.5)),
        'rnc_temperature': tuned.get('rnc_temperature', cfg.get('rnc_temperature', 2.0)),
        'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
        'lambda_deepsad': tuned.get('lambda_deepsad', cfg.get('lambda_deepsad', 1.0)),
        'deepsad_margin': tuned.get('deepsad_margin', cfg.get('deepsad_margin', 1.0)),
        'deepsad_repulsion_weight': cfg.get('deepsad_repulsion_weight', 1.0),
        'lr': tuned.get('lr', cfg.get('lr', 1e-3)),
        'k_per_series': tuned.get('k_per_series', cfg.get('k_per_series', 16)),
        # Transformer-specific (only used when encoder_arch='transformer')
        'd_model': tuned.get('d_model', cfg.get('d_model', 128)),
        'num_heads': tuned.get('num_heads', cfg.get('num_heads', 4)),
        'num_layers': tuned.get('num_layers', cfg.get('num_layers', 4)),
        'ff_dim': tuned.get('ff_dim', cfg.get('ff_dim', 256)),
        'transformer_dropout': tuned.get('transformer_dropout', cfg.get('transformer_dropout', 0.1)),
        'warmup_epochs': tuned.get('warmup_epochs', cfg.get('warmup_epochs', 0)),
    }

    print("\n--- HYPERPARAMS ---")
    for k, v in hp.items():
        print(f"  {k:22s}: {v}")

    center_capture: dict = {}
    model, history, best_epoch = _run_single_training(
        data, hp, device,
        epochs=cfg.get('epochs', 80),
        patience=cfg.get('patience', 15),
        center_out=center_capture,
    )

    # Save model
    model_path = results_dir / 'contrastive_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'hp': hp,
        'input_height': data['input_height'],
        'input_width': data['input_width'],
        'n_in_channels': data['n_in_channels'],
        'best_epoch': best_epoch,
        # Frozen DeepSAD center (z_proj space), if this run used DeepSAD loss —
        # None for SupCon/RNC runs. Needed so downstream diagnostics (e.g. the
        # pooled two-house center check) use the TRUE training-time center
        # instead of a later approximation.
        'deepsad_center': center_capture.get('deepsad_center'),
    }, model_path)
    print(f"\n  Saved model: {model_path}")

    # Persist the NoFlow-only input scaler. Without this, local inference
    # has to refit the scaler from train-NoFlow patches, which is fragile
    # if the local copy of `data/prepared/` ever drifts from the server's.
    scaler_path = results_dir / 'input_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler': data['scaler'],
            'scaler_type': cfg.get('scaler', 'standard'),
            'feature_names': data['feature_names'],
            'n_in_channels': data['n_in_channels'],
            'input_height': data['input_height'],
            'input_width': data['input_width'],
            'cnn_channel_mode': cfg.get('cnn_channel_mode', 'dual'),
        }, f)
    print(f"  Saved input scaler: {scaler_path}")

    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Test evaluation. Four GMM strategies, each answering a different
    # deployment question:
    #   - 'per_series': per-eval-series GMM fitted on that site's own NoFlow.
    #     Optimistic upper bound — assumes recalibration at every new site.
    #   - 'global': single GMM fitted on encoder-train NoFlow, scored against
    #     all eval series. Pessimistic — one calibration, no recalibration.
    #   - 'global_conditional': linear regression z ~ T_pipe on train NoFlow
    #     centroids, single GMM on residuals. Mean-conditioning only —
    #     leftover between-series variance still inflates Σ_residual, so
    #     within-series AUC barely moves (validated: Δ < 0.001 on 18.04).
    #   - 'global_hierarchical': separates centroid model from scatter model.
    #     Per-series-CENTERED residuals (subtract that series' own
    #     centroid) → pooled Σ_within. At eval, centroid is extrapolated
    #     via T-regression. Σ at scoring is purely within-series, no
    #     leftover between-series. Should close the per-site / global gap
    #     for series within the trained T envelope.
    # Aliases: 'both' = per_series + global (back-compat). 'all' = all four.
    eval_mode_raw = cfg.get('gmm_eval_mode', 'per_series')
    _alias = {
        'both': ('per_series', 'global'),
        'all':  ('per_series', 'global', 'global_conditional', 'global_hierarchical'),
    }
    if eval_mode_raw in _alias:
        active_modes = set(_alias[eval_mode_raw])
    elif eval_mode_raw in (
        'per_series', 'global',
        'global_conditional', 'global_hierarchical',
    ):
        active_modes = {eval_mode_raw}
    else:
        print(
            f"  [warn] unknown gmm_eval_mode={eval_mode_raw!r}, "
            f"falling back to 'per_series'."
        )
        active_modes = {'per_series'}

    test_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics_global: Dict[str, Dict[str, float]] = {}
    test_metrics_global_cond: Dict[str, Dict[str, float]] = {}
    test_metrics_global_hier: Dict[str, Dict[str, float]] = {}
    mv_results_global: Dict = {}
    gmm_n = cfg.get('gmm_n_components', 1)
    gmm_cov = cfg.get('gmm_covariance_type', 'full')

    # Build per-patch file-name array for majority voting (mirrors LOSO logic)
    test_files_list: List[str] = []
    for seg_id in test_ids:
        meta = metadata_by_id.get(seg_id)
        if meta is None or 'patch_file' not in meta:
            continue
        patch_path = prepared_base / meta['series'] / meta['patch_file']
        if not patch_path.exists():
            continue
        n = meta.get('n_patches', 0)
        if n == 0:
            continue
        test_files_list.extend([meta['file_name']] * n)
    test_files_arr = np.array(test_files_list) if test_files_list else None

    if 'per_series' in active_modes:
        split_mode = cfg.get('eval_noflow_split_mode', 'fair')
        print(f"\n--- TEST EVALUATION A — per-series GMM (recalibrated per site, mode={split_mode}) ---")
        test_metrics, mv_results, fold_records = _evaluate_on_test_fair(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            test_files=test_files_arr,
            mode=split_mode,
            calibration_file_ratio=cfg.get('eval_calibration_file_ratio', 0.5),
            min_calibration_patches=cfg.get('eval_min_calibration_patches', 500),
        )
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")
        with open(results_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        if mv_results:
            with open(results_dir / 'majority_vote_results.json', 'w') as f:
                json.dump(mv_results, f, indent=2)
            print(f"  Majority vote results saved → majority_vote_results.json")
        if fold_records:
            with open(results_dir / 'test_metrics_fair_folds.json', 'w') as f:
                json.dump(fold_records, f, indent=2)
            print(f"  Per-fold calibration/test audit trail saved → test_metrics_fair_folds.json")

    if 'global' in active_modes:
        print("\n--- TEST EVALUATION B — single global GMM on encoder-train NoFlow ---")
        test_metrics_global, mv_results_global = _evaluate_global_gmm(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            test_files=test_files_arr,
        )
        for k, v in test_metrics_global.items():
            print(f"  {k}: {v}")
        with open(results_dir / 'test_metrics_global.json', 'w') as f:
            json.dump(test_metrics_global, f, indent=2)
        if mv_results_global:
            with open(results_dir / 'majority_vote_results_global.json', 'w') as f:
                json.dump(mv_results_global, f, indent=2)
            print(f"  Majority vote results (global GMM) saved → majority_vote_results_global.json")

    # Both T-conditional modes need per-series temperature map(s); build once
    # if either is active. `gmm_temp_target` may be a single name (legacy,
    # str) or a list of names (e.g. ['temp_pipe', 'temp_amb'] for a joint
    # two-covariate regression) — normalize to a list either way.
    _need_temp = bool({
        'global_conditional', 'global_hierarchical'
    } & active_modes)
    temp_targets_raw = cfg.get('gmm_temp_target', 'temp_pipe')
    temp_targets: List[str] = (
        [temp_targets_raw] if isinstance(temp_targets_raw, str) else list(temp_targets_raw)
    )
    temp_maps: Dict[str, Dict[str, float]] = {}
    if _need_temp:
        unique_series = sorted(
            set(list(data['train_series']) + list(data['test_series']))
        )
        for t in temp_targets:
            temp_maps[t] = _compute_per_series_regress_target(
                data_path, unique_series, t,
            )
            print(f"  Per-series {t}: " + ", ".join(
                f"{s}={temp_maps[t].get(s, float('nan')):.2f}"
                for s in unique_series
            ))

    if 'global_conditional' in active_modes:
        print(
            f"\n--- TEST EVALUATION C — temperature-conditioned global GMM "
            f"({'+'.join(temp_targets)}) ---"
        )
        test_metrics_global_cond, _ = _evaluate_global_gmm_conditional(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            temp_maps=temp_maps,
            temp_target_names=temp_targets,
        )
        for k, v in test_metrics_global_cond.items():
            print(f"  {k}: {v}")
        # Single target keeps the original filename (backward compatible with
        # existing downstream readers); multi-target gets a distinguishing
        # suffix so it doesn't collide with a single-target run's output.
        cond_suffix = '' if len(temp_targets) == 1 else '_' + '_'.join(temp_targets)
        with open(results_dir / f'test_metrics_global_conditional{cond_suffix}.json', 'w') as f:
            json.dump(test_metrics_global_cond, f, indent=2)

    if 'global_hierarchical' in active_modes:
        print(
            "\n--- TEST EVALUATION D — hierarchical global GMM "
            "(per-series-centered Σ + T-extrapolated centroid) ---"
        )
        # Hierarchical mode stays single-covariate — uses the first requested
        # temperature target only.
        test_metrics_global_hier = _evaluate_global_gmm_hierarchical(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            temp_map=temp_maps[temp_targets[0]],
            temp_target_name=temp_targets[0],
        )
        for k, v in test_metrics_global_hier.items():
            print(f"  {k}: {v}")
        with open(results_dir / 'test_metrics_global_hierarchical.json', 'w') as f:
            json.dump(test_metrics_global_hier, f, indent=2)

    # Diagnostics plots
    if plot_enabled and img_out is not None:
        print("\n--- DIAGNOSTIC PLOTS ---")
        _plot_loss_curves(history, img_out / 'loss_curves.png')

        # Reconstructions are only meaningful for the AE variant.
        if _is_autoencoder(encoder_type):
            _plot_reconstructions(
                model, data['val_patches'], data['val_labels'], device,
                img_out / 'reconstructions_val.png',
            )
        else:
            print("  [skip] reconstructions_val.png — not applicable for classifier variant.")

        val_latents = _extract_latents(model, data['val_patches'], device)
        test_latents = _extract_latents(model, data['test_patches'], device)
        all_latents = np.concatenate([val_latents, test_latents], axis=0)
        all_labels = np.concatenate([data['val_labels'], data['test_labels']])
        all_series = list(data['val_series']) + list(data['test_series'])

        _plot_latent_tsne(
            all_latents, all_labels, all_series,
            img_out / 'latent_tsne_by_label.png',
            img_out / 'latent_tsne_by_series.png',
        )

        compactness = _noflow_compactness_per_series(all_latents, all_labels, all_series)
        print("  NoFlow compactness per series:")
        for s, d in compactness.items():
            print(f"    {s}: {d:.4f}")
        with open(results_dir / 'noflow_compactness.json', 'w') as f:
            json.dump(compactness, f, indent=2)

    print("\n" + "=" * 60)
    print("CONTRASTIVE TRAINING COMPLETE")
    print("=" * 60)
    return {
        'history': history,
        'test_metrics': test_metrics,
        'test_metrics_global': test_metrics_global,
        'test_metrics_global_conditional': test_metrics_global_cond,
        'test_metrics_global_hierarchical': test_metrics_global_hier,
        'majority_vote_results_global': mv_results_global,
        'best_epoch': best_epoch,
    }


def eval_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """Load a saved model and run per-series GMM eval + majority voting only (no retraining)."""
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'

    model_path = results_dir / 'contrastive_model.pt'
    scaler_path = results_dir / 'input_scaler.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at {model_path}. Run --run train first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No saved scaler found at {scaler_path}.")

    print("=" * 60)
    print("CONTRASTIVE ENCODER — EVAL ONLY (majority voting)")
    print("=" * 60)

    # Load saved hp and rebuild model
    ckpt = torch.load(model_path, map_location=device)
    hp = ckpt['hp']
    n_in = ckpt['n_in_channels']
    input_height = ckpt['input_height']
    input_width = ckpt['input_width']

    model = build_contrastive_model(
        encoder_type=hp['encoder_type'],
        encoder_arch=hp.get('encoder_arch', 'cnn'),
        in_channels=n_in,
        input_height=input_height,
        input_width=input_width,
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
    print(f"  Loaded model from {model_path} (best_epoch={ckpt.get('best_epoch', '?')})")

    # Load saved scaler
    with open(scaler_path, 'rb') as f:
        scaler_bundle = pickle.load(f)
    saved_scaler = scaler_bundle['scaler']
    feat_names = scaler_bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    dual = (scaler_bundle.get('cnn_channel_mode', 'dual') == 'dual')

    # Resolve test split
    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)
    encoder_train_series = cfg.get('encoder_train_series') or None
    gmm_eval_series = cfg.get('gmm_eval_series') or None
    train_ids, val_ids, test_ids, train_resolved, eval_resolved = _resolve_train_eval_ids(
        splits_data, metadata_by_id, encoder_train_series, gmm_eval_series,
    )
    print(f"  encoder_train_series: {train_resolved}")
    print(f"  gmm_eval_series:      {eval_resolved}")

    # Load and scale test patches using saved scaler
    selected = _resolve_selected_features(cfg)
    te_p, te_l, te_s, _ = _prepare_patches(test_ids, prepared_base, metadata_by_id, selected)
    if saved_scaler is not None:
        n_, f_, w_ = te_p.shape
        flat_ = te_p.transpose(0, 2, 1).reshape(-1, f_)
        te_p = saved_scaler.transform(flat_).reshape(n_, w_, f_).transpose(0, 2, 1).astype(np.float32)
    te_c, _ = _to_channels(te_p, te_p.shape[1], te_p.shape[2], n_stats, dual)

    # Determine which GMM evaluation modes to run.
    eval_mode_raw = cfg.get('gmm_eval_mode', 'per_series')
    _eval_alias = {
        'both': ('per_series', 'global'),
        'all':  ('per_series', 'global', 'global_conditional', 'global_hierarchical'),
    }
    if eval_mode_raw in _eval_alias:
        active_modes = set(_eval_alias[eval_mode_raw])
    elif eval_mode_raw in ('per_series', 'global', 'global_conditional', 'global_hierarchical'):
        active_modes = {eval_mode_raw}
    else:
        print(f"  [warn] unknown gmm_eval_mode={eval_mode_raw!r}, falling back to 'per_series'.")
        active_modes = {'per_series'}

    # Load encoder-train patches when any global-GMM mode is requested.
    need_train = bool({'global', 'global_conditional', 'global_hierarchical'} & active_modes)
    tr_c: np.ndarray = np.empty((0,), dtype=np.float32)
    tr_l: np.ndarray = np.array([], dtype=np.int64)
    tr_s: List[str] = []
    if need_train:
        tr_p_raw, tr_l, tr_s, _ = _prepare_patches(
            train_ids + val_ids, prepared_base, metadata_by_id, selected,
        )
        if saved_scaler is not None and len(tr_p_raw) > 0:
            n_, f_, w_ = tr_p_raw.shape
            flat_ = tr_p_raw.transpose(0, 2, 1).reshape(-1, f_)
            tr_p_raw = saved_scaler.transform(flat_).reshape(n_, w_, f_).transpose(0, 2, 1).astype(np.float32)
        f_dim = tr_p_raw.shape[1] if len(tr_p_raw) > 0 else (te_p.shape[1] if len(te_p) > 0 else n_in * input_height)
        w_dim = tr_p_raw.shape[2] if len(tr_p_raw) > 0 else (te_p.shape[2] if len(te_p) > 0 else input_width)
        tr_c, _ = _to_channels(tr_p_raw, f_dim, w_dim, n_stats, dual)
        print(f"  Loaded {len(tr_c)} encoder-train patches for global GMM.")

    data = {
        'train_patches': tr_c,
        'train_labels': tr_l,
        'train_series': tr_s,
        'test_patches': te_c,
        'test_labels': te_l,
        'test_series': te_s,
    }

    # Build per-patch file array for majority voting
    test_files_list: List[str] = []
    for seg_id in test_ids:
        meta = metadata_by_id.get(seg_id)
        if meta is None or 'patch_file' not in meta:
            continue
        patch_path = prepared_base / meta['series'] / meta['patch_file']
        if not patch_path.exists():
            continue
        n = meta.get('n_patches', 0)
        if n == 0:
            continue
        test_files_list.extend([meta['file_name']] * n)
    test_files_arr = np.array(test_files_list) if test_files_list else None

    gmm_n = cfg.get('gmm_n_components', 1)
    gmm_cov = cfg.get('gmm_covariance_type', 'full')

    test_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics_global: Dict[str, Dict[str, float]] = {}
    mv_results: Dict = {}

    if 'per_series' in active_modes:
        split_mode = cfg.get('eval_noflow_split_mode', 'fair')
        print(f"\n--- EVAL — per-series GMM + majority voting (mode={split_mode}) ---")
        test_metrics, mv_results, fold_records = _evaluate_on_test_fair(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            test_files=test_files_arr,
            mode=split_mode,
            calibration_file_ratio=cfg.get('eval_calibration_file_ratio', 0.5),
            min_calibration_patches=cfg.get('eval_min_calibration_patches', 500),
        )
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")
        with open(results_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        if mv_results:
            mv_path = results_dir / 'majority_vote_results.json'
            with open(mv_path, 'w') as f:
                json.dump(mv_results, f, indent=2)
            print(f"\nMajority vote results:")
            for series, files in mv_results.items():
                correct = sum(1 for v in files.values() if v['true_label'] == v['majority_pred'])
                print(f"  {series}: {correct}/{len(files)} files correct")
            print(f"  Saved → {mv_path}")
        if fold_records:
            fold_path = results_dir / 'test_metrics_fair_folds.json'
            with open(fold_path, 'w') as f:
                json.dump(fold_records, f, indent=2)
            print(f"  Per-fold calibration/test audit trail saved → {fold_path}")

    mv_results_global: Dict = {}
    if 'global' in active_modes:
        print("\n--- EVAL — single global GMM on encoder-train NoFlow ---")
        test_metrics_global, mv_results_global = _evaluate_global_gmm(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            test_files=test_files_arr,
        )
        for k, v in test_metrics_global.items():
            print(f"  {k}: {v}")
        with open(results_dir / 'test_metrics_global.json', 'w') as f:
            json.dump(test_metrics_global, f, indent=2)
        if mv_results_global:
            mv_path_global = results_dir / 'majority_vote_results_global.json'
            with open(mv_path_global, 'w') as f:
                json.dump(mv_results_global, f, indent=2)
            print(f"  Majority vote results (global GMM) saved → {mv_path_global}")

    test_metrics_global_cond: Dict[str, Dict[str, float]] = {}
    if 'global_conditional' in active_modes:
        # gmm_temp_target may be a single name (legacy, str) or a list of
        # names (e.g. ['temp_pipe', 'temp_amb'] for a joint two-covariate
        # regression) — normalize to a list either way.
        temp_targets_raw = cfg.get('gmm_temp_target', 'temp_pipe')
        temp_targets: List[str] = (
            [temp_targets_raw] if isinstance(temp_targets_raw, str) else list(temp_targets_raw)
        )
        unique_series = sorted(set(list(data['train_series']) + list(data['test_series'])))
        temp_maps: Dict[str, Dict[str, float]] = {}
        for t in temp_targets:
            temp_maps[t] = _compute_per_series_regress_target(data_path, unique_series, t)
            print(f"  Per-series {t}: " + ", ".join(
                f"{s}={temp_maps[t].get(s, float('nan')):.2f}" for s in unique_series
            ))

        print(
            f"\n--- EVAL — temperature-conditioned global GMM "
            f"({'+'.join(temp_targets)}) ---"
        )
        test_metrics_global_cond, _ = _evaluate_global_gmm_conditional(
            model, data, device,
            gmm_n_components=gmm_n,
            gmm_covariance_type=gmm_cov,
            temp_maps=temp_maps,
            temp_target_names=temp_targets,
        )
        for k, v in test_metrics_global_cond.items():
            print(f"  {k}: {v}")
        # Single target keeps the original filename (backward compatible);
        # multi-target gets a distinguishing suffix.
        cond_suffix = '' if len(temp_targets) == 1 else '_' + '_'.join(temp_targets)
        with open(results_dir / f'test_metrics_global_conditional{cond_suffix}.json', 'w') as f:
            json.dump(test_metrics_global_cond, f, indent=2)

    return {
        'test_metrics': test_metrics,
        'test_metrics_global': test_metrics_global,
        'test_metrics_global_conditional': test_metrics_global_cond,
        'majority_vote_results': mv_results,
        'majority_vote_results_global': mv_results_global,
    }


def predict_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
    as_blind_series: Optional[List[str]] = None,
) -> Dict:
    """Blind inference: load saved model, fit global GMM on encoder-train NoFlow,
    score predict-series patches without requiring ground-truth labels.

    Series with "blind": true in label.json land in the blind_test bucket in
    splits.json and are picked up automatically via --gmm-eval-series.

    --as-blind-series treats any already-labeled series as blind (sanity test):
    prediction runs label-free and the output CSV additionally carries true_class
    for direct comparison. The series must NOT be in encoder_train_series.

    Outputs:
      blind_predictions_<series>.csv  — one row per file with predicted_class_vote,
        predicted_class_meanp, p_leakage_mean, p_leakage_max, n_patches, [true_class]
      predict_scores.json             — raw per-patch scores (legacy, kept for compat)
    """
    import csv as _csv

    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'

    model_path = results_dir / 'contrastive_model.pt'
    scaler_path = results_dir / 'input_scaler.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at {model_path}. Run --run train first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No saved scaler found at {scaler_path}.")

    print("=" * 60)
    print("CONTRASTIVE ENCODER — PREDICT (blind scoring)")
    print("=" * 60)

    ckpt = torch.load(model_path, map_location=device)
    hp = ckpt['hp']
    n_in = ckpt['n_in_channels']
    input_height = ckpt['input_height']
    input_width = ckpt['input_width']

    model = build_contrastive_model(
        encoder_type=hp['encoder_type'],
        encoder_arch=hp.get('encoder_arch', 'cnn'),
        in_channels=n_in,
        input_height=input_height,
        input_width=input_width,
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
    print(f"  Loaded model from {model_path} (best_epoch={ckpt.get('best_epoch', '?')})")

    with open(scaler_path, 'rb') as f:
        scaler_bundle = pickle.load(f)
    saved_scaler = scaler_bundle['scaler']
    feat_names = scaler_bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    dual = (scaler_bundle.get('cnn_channel_mode', 'dual') == 'dual')

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)
    encoder_train_series = cfg.get('encoder_train_series') or None
    predict_series = cfg.get('gmm_eval_series') or None
    as_blind = list(as_blind_series) if as_blind_series else []

    if not predict_series and not as_blind:
        raise ValueError(
            "--gmm-eval-series must be provided for predict mode "
            "(specifies which series to score)."
        )

    all_predict_series = list(predict_series or []) + [s for s in as_blind if s not in (predict_series or [])]

    train_ids, val_ids, _, train_resolved, _ = _resolve_train_eval_ids(
        splits_data, metadata_by_id, encoder_train_series, None,
    )
    predict_ids = _ids_for_blind_series(splits_data, metadata_by_id, all_predict_series)

    print(f"  encoder_train_series: {train_resolved}")
    print(f"  predict_series:       {sorted(all_predict_series)}")
    print(f"  as_blind_series:      {sorted(as_blind) if as_blind else '(none)'}")
    print(f"  predict segments:     {len(predict_ids)}")

    selected = _resolve_selected_features(cfg)

    def _apply_scaler(patches: np.ndarray) -> np.ndarray:
        if saved_scaler is None or len(patches) == 0:
            return patches
        n_, f_, w_ = patches.shape
        flat_ = patches.transpose(0, 2, 1).reshape(-1, f_)
        return saved_scaler.transform(flat_).reshape(n_, w_, f_).transpose(0, 2, 1).astype(np.float32)

    # Load encoder-train patches to fit the global GMM.
    tr_p, tr_l, _, _ = _prepare_patches(train_ids + val_ids, prepared_base, metadata_by_id, selected)
    tr_p = _apply_scaler(tr_p)
    if len(tr_p) == 0:
        raise RuntimeError("No encoder-train patches found — cannot fit global GMM.")
    f_dim, w_dim = tr_p.shape[1], tr_p.shape[2]
    tr_c, _ = _to_channels(tr_p, f_dim, w_dim, n_stats, dual)

    # Load predict-series patches (all of them — no label filtering at inference).
    if not predict_ids:
        print("  [warn] No segments found for predict series. Was preparation run?")
        return {}
    pr_p, pr_l, pr_s, _ = _prepare_patches(predict_ids, prepared_base, metadata_by_id, selected)
    pr_p = _apply_scaler(pr_p)
    if len(pr_p) == 0:
        print("  [warn] No patches found for predict series. Check preparation output.")
        return {}
    pr_c, _ = _to_channels(pr_p, f_dim, w_dim, n_stats, dual)

    # Build per-patch file mapping (and optional true-label for --as-blind-series).
    predict_files_list: List[str] = []
    predict_true_labels: List[int] = []
    for seg_id in predict_ids:
        meta = metadata_by_id.get(seg_id)
        if meta is None or 'patch_file' not in meta:
            continue
        patch_path = prepared_base / meta['series'] / meta['patch_file']
        if not patch_path.exists():
            continue
        n = meta.get('n_patches', 0)
        predict_files_list.extend([meta['file_name']] * n)
        predict_true_labels.extend([meta.get('label', -1)] * n)
    predict_files_arr = np.array(predict_files_list) if predict_files_list else None
    true_labels_arr = np.array(predict_true_labels) if predict_true_labels else None

    # Fit global GMM on encoder-train NoFlow latents.
    print(f"\n  Fitting global GMM on {int((np.asarray(tr_l) == 0).sum())} train-NoFlow latents...")
    train_latents = _extract_latents(model, tr_c, device)
    train_noflow_mask = np.asarray(tr_l) == 0
    if train_noflow_mask.sum() < 4:
        raise RuntimeError("Not enough train-NoFlow patches to fit GMM.")
    gmm_n = cfg.get('gmm_n_components', 1)
    gmm_cov = cfg.get('gmm_covariance_type', 'full')
    gmm = GaussianMixture(
        n_components=gmm_n,
        covariance_type=gmm_cov,
        reg_covar=1e-3,
        random_state=42,
    )
    gmm.fit(train_latents[train_noflow_mask])

    # Compute threshold from training-NoFlow score distribution (Gaussian, sigma_k=3).
    train_nf_scores = -gmm.score_samples(train_latents[train_noflow_mask])
    thr = float(train_nf_scores.mean() + 3.0 * train_nf_scores.std())
    sorted_nf_scores = np.sort(train_nf_scores)  # for ECDF

    def _ecdf(score: float) -> float:
        """Fraction of training-NoFlow patches with score <= this score."""
        return float(np.searchsorted(sorted_nf_scores, score, side='right') / len(sorted_nf_scores))

    # Score all predict patches — higher score = more anomalous.
    print(f"  Scoring {len(pr_c)} predict patches (threshold={thr:.3f})...")
    predict_latents = _extract_latents(model, pr_c, device)
    scores = -gmm.score_samples(predict_latents)

    # Aggregate per file.
    per_file: Dict[str, Dict] = {}
    if predict_files_arr is not None and len(predict_files_arr) == len(scores):
        patch_preds = (scores >= thr).astype(int)
        for fname in sorted(set(predict_files_arr)):
            mask = predict_files_arr == fname
            s = scores[mask]
            preds = patch_preds[mask]
            n_lk = int(preds.sum())
            n_nf = int((preds == 0).sum())
            p_mean = _ecdf(float(s.mean()))
            p_max = _ecdf(float(s.max()))
            vote = 'LEAKAGE' if n_lk > n_nf else 'NOFLOW'
            meanp = 'LEAKAGE' if p_mean > 0.95 else 'NOFLOW'
            entry: Dict = {
                'series': str(pr_s[np.where(mask)[0][0]]) if len(pr_s) > 0 else '',
                'n_patches': int(mask.sum()),
                'predicted_class_vote': vote,
                'predicted_class_meanp': meanp,
                'p_leakage_mean': round(p_mean, 4),
                'p_leakage_max': round(p_max, 4),
                'n_patches_leakage': n_lk,
                'n_patches_noflow': n_nf,
                'score_mean': float(s.mean()),
                'score_max': float(s.max()),
                'score_p95': float(np.percentile(s, 95)),
            }
            # True label for --as-blind-series sanity mode
            if true_labels_arr is not None and fname in as_blind:
                file_labels = true_labels_arr[mask]
                entry['true_class'] = 'LEAKAGE' if (file_labels == 1).any() else 'NOFLOW'
            per_file[fname] = entry

    # Write blind_predictions_<series>.csv per predict series.
    csv_paths: List[str] = []
    for series_name in sorted(all_predict_series):
        rows = {k: v for k, v in per_file.items() if v.get('series') == series_name}
        if not rows:
            continue
        safe_name = series_name.replace('/', '_').replace(' ', '_')
        csv_path = results_dir / f'blind_predictions_{safe_name}.csv'
        has_true = any('true_class' in r for r in rows.values())
        fieldnames = [
            'file', 'predicted_class_vote', 'predicted_class_meanp',
            'p_leakage_mean', 'p_leakage_max',
            'n_patches', 'n_patches_leakage', 'n_patches_noflow',
            'score_mean', 'score_max', 'score_p95',
        ]
        if has_true:
            fieldnames.append('true_class')
        with open(csv_path, 'w', newline='') as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for fname in sorted(rows):
                writer.writerow({'file': fname, **rows[fname]})
        csv_paths.append(str(csv_path))
        print(f"\n  Blind predictions — {series_name}:")
        for fname in sorted(rows):
            r = rows[fname]
            tc = f"  true={r['true_class']}" if 'true_class' in r else ''
            print(f"    {fname}: vote={r['predicted_class_vote']}  "
                  f"p_mean={r['p_leakage_mean']:.3f}  p_max={r['p_leakage_max']:.3f}{tc}")
        print(f"  Saved → {csv_path}")

    # Cluster stats for adaptation diagnostic — compactness + GMM score per predict series.
    # Interpretation: if noflow_compactness is similar to training reference and trace_cov is
    # not pathologically large, a per-site GMM refit is sufficient for adaptation.
    # If compactness >> training reference (>5×), the encoder can't represent the series
    # consistently → a projection-head alignment step is needed first.
    print("\n  --- Cluster stats (adaptation diagnostic) ---")
    pr_s_arr = np.array(pr_s)
    cluster_stats: Dict[str, dict] = {}
    for sname in sorted(set(pr_s)):
        mask_s = pr_s_arr == sname
        z_s = predict_latents[mask_s]
        if len(z_s) < 4:
            continue
        centroid = z_s.mean(axis=0)
        dists = np.linalg.norm(z_s - centroid, axis=1)
        cov_mat = np.cov(z_s.T)
        trace_cov = float(np.trace(cov_mat)) if cov_mat.ndim == 2 else float(cov_mat)
        cluster_stats[sname] = {
            'n_patches': int(mask_s.sum()),
            'noflow_compactness': round(float(dists.mean()), 4),
            'noflow_compactness_std': round(float(dists.std()), 4),
            'trace_cov': float(trace_cov),
            'centroid_norm': round(float(np.linalg.norm(centroid)), 4),
            'global_gmm_score_mean': round(float(scores[mask_s].mean()), 4),
        }
        print(f"    {sname}: compactness={dists.mean():.2f}  trace_cov={trace_cov:.2e}"
              f"  gmm_score={scores[mask_s].mean():.1f}")

    # Training-NoFlow reference for comparison.
    tr_nf_z = train_latents[train_noflow_mask]
    tr_nf_c = tr_nf_z.mean(axis=0)
    tr_nf_dists = np.linalg.norm(tr_nf_z - tr_nf_c, axis=1)
    tr_cov = np.cov(tr_nf_z.T)
    cluster_stats['_training_noflow_reference'] = {
        'n_patches': int(train_noflow_mask.sum()),
        'noflow_compactness': round(float(tr_nf_dists.mean()), 4),
        'noflow_compactness_std': round(float(tr_nf_dists.std()), 4),
        'trace_cov': float(np.trace(tr_cov)) if tr_cov.ndim == 2 else float(tr_cov),
        'centroid_norm': round(float(np.linalg.norm(tr_nf_c)), 4),
    }
    print(f"    [training NoFlow ref]: compactness={tr_nf_dists.mean():.2f}"
          f"  trace_cov={np.trace(tr_cov) if tr_cov.ndim == 2 else tr_cov:.2e}")

    cluster_stats_path = results_dir / 'blind_cluster_stats.json'
    with open(cluster_stats_path, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    print(f"  blind_cluster_stats.json → {cluster_stats_path}")

    # Per-site GMM refit on predict-series NoFlow latents.
    # Fits one GMM per series, then cross-scores each series against every other
    # series' GMM. If cross-scores are similar to own-scores, the per-site GMM
    # generalises within the house — validating the refit approach before leakage
    # data arrives.
    print("\n  --- Per-site GMM refit (cross-series validation) ---")
    per_site_gmms: Dict[str, GaussianMixture] = {}
    per_site_scores: Dict[str, Dict] = {}
    for sname in sorted(set(pr_s)):
        mask_s = pr_s_arr == sname
        z_s = predict_latents[mask_s]
        if len(z_s) < 4:
            continue
        gmm_site = GaussianMixture(
            n_components=1, covariance_type='full', reg_covar=1e-3, random_state=42
        )
        gmm_site.fit(z_s)
        per_site_gmms[sname] = gmm_site
        own_s = -gmm_site.score_samples(z_s)
        thr_site = float(own_s.mean() + 3.0 * own_s.std())
        per_site_scores[sname] = {
            'n_patches': int(len(z_s)),
            'own_score_mean': round(float(own_s.mean()), 4),
            'own_score_std': round(float(own_s.std()), 4),
            'own_threshold': round(thr_site, 4),
        }
        print(f"    {sname}: own score mean={own_s.mean():.3f}  std={own_s.std():.3f}"
              f"  threshold={thr_site:.3f}")

    # Cross-scores: score each series against every other series' GMM.
    cross_results: Dict[str, Dict] = {}
    if len(per_site_gmms) > 1:
        print("    Cross-scores (does hb-A GMM accept hb-B NoFlow as inlier?):")
        for src, gmm_src in per_site_gmms.items():
            thr_src = per_site_scores[src]['own_threshold']
            for tgt in sorted(per_site_gmms):
                if src == tgt:
                    continue
                mask_tgt = pr_s_arr == tgt
                z_tgt = predict_latents[mask_tgt]
                cs = -gmm_src.score_samples(z_tgt)
                pct_inlier = float((cs < thr_src).mean())
                cross_results[f'{src}_scores_{tgt}'] = {
                    'score_mean': round(float(cs.mean()), 4),
                    'score_std': round(float(cs.std()), 4),
                    'pct_inlier': round(pct_inlier, 4),
                }
                status = 'INLIER ✓' if pct_inlier > 0.9 else ('PARTIAL' if pct_inlier > 0.5 else 'OUTLIER ✗')
                print(f"      GMM({src}) → {tgt}: mean={cs.mean():.3f}  "
                      f"pct_inlier={pct_inlier:.1%}  [{status}]")

    # Save per-site GMMs so they can be reloaded when leakage data arrives.
    gmm_pkl_path = results_dir / 'blind_per_site_gmms.pkl'
    with open(gmm_pkl_path, 'wb') as f:
        import pickle as _pkl
        _pkl.dump({'gmms': per_site_gmms, 'scores': per_site_scores,
                   'cross': cross_results}, f)
    print(f"  blind_per_site_gmms.pkl → {gmm_pkl_path}")

    out = {
        'predict_series': sorted(all_predict_series),
        'encoder_train_series': train_resolved,
        'n_patches_total': int(len(scores)),
        'gmm_fit_n_noflow': int(train_noflow_mask.sum()),
        'threshold': thr,
        'score_mean': float(scores.mean()),
        'score_max': float(scores.max()),
        'score_p95': float(np.percentile(scores, 95)),
        'per_file': {k: {kk: vv for kk, vv in v.items() if kk != 'series'} for k, v in per_file.items()},
        'csv_paths': csv_paths,
    }

    pred_path = results_dir / 'predict_scores.json'
    with open(pred_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  predict_scores.json → {pred_path}")

    return out


# ===========================================================================
# Detector comparison: Mahalanobis / SVDD / GMM / kNN on z_proj
# ===========================================================================

def _mahalanobis_detector(
    train_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Mahalanobis detector on train_z (NoFlow).
    Returns (mean, precision_matrix). Precision = inverse of regularised covariance."""
    mu = train_z.mean(axis=0)
    cov = np.cov(train_z.T)
    # Regularise: add small diagonal to avoid singular matrix.
    cov += np.eye(cov.shape[0]) * 1e-6
    precision = np.linalg.inv(cov)
    return mu, precision


def _mahalanobis_score(z: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Mahalanobis distance for each row of z. Higher = more anomalous."""
    diff = z - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, precision, diff))


def _append_temp(
    latents: np.ndarray,
    series_arr: np.ndarray,
    series_temps: Dict[str, float],
) -> np.ndarray:
    """Append per-series mean temp_pipe as an extra column to latents.
    Series with no temperature entry are given the global mean."""
    temps = np.array([series_temps.get(s, float('nan')) for s in series_arr])
    all_known = temps[~np.isnan(temps)]
    global_mean = float(all_known.mean()) if len(all_known) > 0 else 0.0
    temps = np.where(np.isnan(temps), global_mean, temps)
    return np.concatenate([latents, temps[:, None]], axis=1)


def detector_eval_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """Compare anomaly detectors applied to z_proj (L2-normalised projection head output).

    Step 1 — Mahalanobis distance on z_proj, with and without temp_pipe appended.
    Later steps will add SVDD, GMM on z_proj, kNN.

    For each detector × feature variant:
      - Fit on encoder-train NoFlow z_proj
      - Score eval series
      - Compute per-series ROC-AUC
      - Write detector_eval_results.json

    Uses same train/eval split as --run eval (encoder_train_series / gmm_eval_series).
    """
    from sklearn.metrics import roc_auc_score

    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'

    model_path = results_dir / 'contrastive_model.pt'
    scaler_path = results_dir / 'input_scaler.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Run --run train first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No saved scaler at {scaler_path}.")

    print("=" * 60)
    print("CONTRASTIVE ENCODER — DETECTOR EVAL (z_proj)")
    print("=" * 60)

    ckpt = torch.load(model_path, map_location=device)
    hp = ckpt['hp']
    n_in = ckpt['n_in_channels']
    input_height = ckpt['input_height']
    input_width = ckpt['input_width']

    model = build_contrastive_model(
        encoder_type=hp['encoder_type'],
        encoder_arch=hp.get('encoder_arch', 'cnn'),
        in_channels=n_in,
        input_height=input_height,
        input_width=input_width,
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
    print(f"  Loaded model (best_epoch={ckpt.get('best_epoch', '?')},"
          f" latent_dim={hp['latent_dim']}, projection_dim={hp['projection_dim']})")

    with open(scaler_path, 'rb') as f:
        scaler_bundle = pickle.load(f)
    saved_scaler = scaler_bundle['scaler']
    feat_names = scaler_bundle.get('feature_names', [])
    n_stats = detect_n_stats(feat_names) if feat_names else 1
    dual = (scaler_bundle.get('cnn_channel_mode', 'dual') == 'dual')

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)
    encoder_train_series = cfg.get('encoder_train_series') or None
    gmm_eval_series = cfg.get('gmm_eval_series') or None

    train_ids, val_ids, eval_ids, train_resolved, eval_resolved = _resolve_train_eval_ids(
        splits_data, metadata_by_id, encoder_train_series, gmm_eval_series,
    )
    print(f"  encoder_train_series: {train_resolved}")
    print(f"  gmm_eval_series:      {eval_resolved}")

    selected = _resolve_selected_features(cfg)

    def _apply_scaler(patches: np.ndarray) -> np.ndarray:
        if saved_scaler is None or len(patches) == 0:
            return patches
        n_, f_, w_ = patches.shape
        flat_ = patches.transpose(0, 2, 1).reshape(-1, f_)
        return saved_scaler.transform(flat_).reshape(n_, w_, f_).transpose(0, 2, 1).astype(np.float32)

    # Load patches.
    tr_p, tr_l, tr_s, _ = _prepare_patches(train_ids + val_ids, prepared_base, metadata_by_id, selected)
    tr_p = _apply_scaler(tr_p)
    ev_p, ev_l, ev_s, _ = _prepare_patches(eval_ids, prepared_base, metadata_by_id, selected)
    ev_p = _apply_scaler(ev_p)

    f_dim, w_dim = tr_p.shape[1], tr_p.shape[2]
    tr_c, _ = _to_channels(tr_p, f_dim, w_dim, n_stats, dual)
    ev_c, _ = _to_channels(ev_p, f_dim, w_dim, n_stats, dual)

    # Extract z_proj for train and eval.
    print("\n  Extracting z_proj (projection head output)...")
    tr_zp = _extract_latents(model, tr_c, device, use_proj=True)
    ev_zp = _extract_latents(model, ev_c, device, use_proj=True)
    tr_l_arr = np.asarray(tr_l)
    ev_l_arr = np.asarray(ev_l)
    ev_s_arr = np.asarray(ev_s)

    train_noflow_mask = tr_l_arr == 0
    print(f"  train NoFlow z_proj: {train_noflow_mask.sum()} patches  "
          f"(projection_dim={tr_zp.shape[1]})")

    # Load per-series mean temp_pipe for temperature-augmented variants.
    all_series = list(set(list(train_resolved) + list(eval_resolved)))
    series_temps = _compute_per_series_regress_target(data_path, all_series, target_name='temp_pipe')
    tr_s_arr = np.asarray(tr_s)

    # --- STEP 1: Mahalanobis ---
    print("\n  === Step 1: Mahalanobis distance on z_proj ===")
    results: Dict[str, Dict] = {}

    for use_temp, label_suffix in [(False, 'z_proj'), (True, 'z_proj+temp')]:
        variant = f'mahalanobis_{label_suffix}'
        print(f"\n  Variant: {variant}")

        # Build feature matrices.
        if use_temp:
            tr_feat = _append_temp(tr_zp, tr_s_arr, series_temps)
            ev_feat = _append_temp(ev_zp, ev_s_arr, series_temps)
        else:
            tr_feat = tr_zp
            ev_feat = ev_zp

        # Fit on train NoFlow.
        tr_nf_feat = tr_feat[train_noflow_mask]
        mu, precision = _mahalanobis_detector(tr_nf_feat)

        # Score all eval patches.
        ev_scores = _mahalanobis_score(ev_feat, mu, precision)

        # Per-series AUC.
        series_aucs: Dict[str, float] = {}
        for sname in sorted(set(ev_s)):
            mask_s = ev_s_arr == sname
            y_true = ev_l_arr[mask_s]
            y_score = ev_scores[mask_s]
            if len(np.unique(y_true)) < 2:
                print(f"    {sname}: skipped (only one class)")
                continue
            auc = roc_auc_score(y_true, y_score)
            series_aucs[sname] = round(float(auc), 4)
            print(f"    {sname}: AUC={auc:.4f}  "
                  f"(NoFlow score mean={y_score[y_true==0].mean():.1f},"
                  f" Flow score mean={y_score[y_true==1].mean():.1f})")

        mean_auc = round(float(np.mean(list(series_aucs.values()))), 4) if series_aucs else float('nan')
        print(f"    → mean AUC = {mean_auc:.4f}")
        results[variant] = {'series_aucs': series_aucs, 'mean_auc': mean_auc}

    # --- Step 2: SVDD (OneClassSVM, RBF kernel) on z_proj ---
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler as _SS

    print("\n  --- SVDD (OneClassSVM, RBF) ---")
    for use_temp, label_suffix in [(False, 'z_proj'), (True, 'z_proj+temp')]:
        variant = f'svdd_{label_suffix}'
        print(f"\n  Variant: {variant}")

        if use_temp:
            tr_feat = _append_temp(tr_zp, tr_s_arr, series_temps)
            ev_feat = _append_temp(ev_zp, ev_s_arr, series_temps)
        else:
            tr_feat = tr_zp
            ev_feat = ev_zp

        tr_nf_feat = tr_feat[train_noflow_mask]

        # Scale before SVM (RBF kernel is sensitive to feature scale).
        scaler = _SS()
        tr_nf_scaled = scaler.fit_transform(tr_nf_feat)
        ev_scaled = scaler.transform(ev_feat)

        ocsvm = OneClassSVM(kernel='rbf', nu=0.05)
        ocsvm.fit(tr_nf_scaled)
        # Negative decision function: higher = more anomalous.
        ev_scores = -ocsvm.decision_function(ev_scaled)

        series_aucs: Dict[str, float] = {}
        for sname in sorted(set(ev_s)):
            mask_s = ev_s_arr == sname
            y_true = ev_l_arr[mask_s]
            y_score = ev_scores[mask_s]
            if len(np.unique(y_true)) < 2:
                print(f"    {sname}: skipped (only one class)")
                continue
            auc = roc_auc_score(y_true, y_score)
            series_aucs[sname] = round(float(auc), 4)
            nf_mean = y_score[y_true == 0].mean()
            fl_mean = y_score[y_true == 1].mean()
            print(f"    {sname}: AUC={auc:.4f}  "
                  f"(NoFlow score mean={nf_mean:.3f},"
                  f" Flow score mean={fl_mean:.3f})")

        mean_auc = round(float(np.mean(list(series_aucs.values()))), 4) if series_aucs else float('nan')
        print(f"    → mean AUC = {mean_auc:.4f}")
        results[variant] = {'series_aucs': series_aucs, 'mean_auc': mean_auc}

    # --- Haus B cross-temporal diagnostic ---
    # Fit each detector on hb-26.04 NoFlow z_proj, score hb-4.05 NoFlow z_proj.
    # Reports inlier rate: % of hb-4.05 patches below the 95th-pct threshold of hb-26.04 scores.
    # Answers: which detector best bridges the 5-day temporal gap within Haus B?
    HB_REF = 'hb-26.04.2026'
    HB_TEST = 'hb-4.05.2026'
    hb_ref_mask = ev_s_arr == HB_REF
    hb_test_mask = ev_s_arr == HB_TEST
    if hb_ref_mask.sum() > 0 and hb_test_mask.sum() > 0:
        print(f"\n  === Haus B cross-temporal diagnostic ({HB_REF} → {HB_TEST}) ===")
        print(f"  Reference NoFlow patches: {hb_ref_mask.sum()}  "
              f"| Test NoFlow patches: {hb_test_mask.sum()}")
        hb_ref_zp = ev_zp[hb_ref_mask]   # hb-26.04 NoFlow z_proj
        hb_test_zp = ev_zp[hb_test_mask]  # hb-4.05 NoFlow z_proj

        hb_results: Dict[str, Dict] = {}

        # Mahalanobis
        mu_hb, prec_hb = _mahalanobis_detector(hb_ref_zp)
        ref_scores_mh = _mahalanobis_score(hb_ref_zp, mu_hb, prec_hb)
        test_scores_mh = _mahalanobis_score(hb_test_zp, mu_hb, prec_hb)
        thresh_mh = float(np.percentile(ref_scores_mh, 95))
        inlier_rate_mh = float((test_scores_mh <= thresh_mh).mean())
        print(f"\n  Mahalanobis:")
        print(f"    hb-26.04 score mean={ref_scores_mh.mean():.2f}  "
              f"p95 threshold={thresh_mh:.2f}")
        print(f"    hb-4.05  score mean={test_scores_mh.mean():.2f}  "
              f"inlier_rate={inlier_rate_mh:.2%}")
        hb_results['mahalanobis_z_proj'] = {
            'ref_score_mean': round(float(ref_scores_mh.mean()), 4),
            'test_score_mean': round(float(test_scores_mh.mean()), 4),
            'threshold_p95': round(thresh_mh, 4),
            'inlier_rate': round(inlier_rate_mh, 4),
        }

        # SVDD
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler as _SS2
        scaler_hb = _SS2()
        hb_ref_scaled = scaler_hb.fit_transform(hb_ref_zp)
        hb_test_scaled = scaler_hb.transform(hb_test_zp)
        ocsvm_hb = OneClassSVM(kernel='rbf', nu=0.05)
        ocsvm_hb.fit(hb_ref_scaled)
        ref_scores_sv = -ocsvm_hb.decision_function(hb_ref_scaled)
        test_scores_sv = -ocsvm_hb.decision_function(hb_test_scaled)
        thresh_sv = float(np.percentile(ref_scores_sv, 95))
        inlier_rate_sv = float((test_scores_sv <= thresh_sv).mean())
        print(f"\n  SVDD (OneClassSVM RBF):")
        print(f"    hb-26.04 score mean={ref_scores_sv.mean():.4f}  "
              f"p95 threshold={thresh_sv:.4f}")
        print(f"    hb-4.05  score mean={test_scores_sv.mean():.4f}  "
              f"inlier_rate={inlier_rate_sv:.2%}")
        hb_results['svdd_z_proj'] = {
            'ref_score_mean': round(float(ref_scores_sv.mean()), 4),
            'test_score_mean': round(float(test_scores_sv.mean()), 4),
            'threshold_p95': round(thresh_sv, 4),
            'inlier_rate': round(inlier_rate_sv, 4),
        }

        results['hausb_cross_temporal'] = hb_results
        print(f"\n  Haus B summary (inlier rate, higher = more robust to 5-day drift):")
        for det_name, det_r in hb_results.items():
            print(f"    {det_name:<30} inlier_rate={det_r['inlier_rate']:.2%}")
    else:
        missing = []
        if hb_ref_mask.sum() == 0:
            missing.append(HB_REF)
        if hb_test_mask.sum() == 0:
            missing.append(HB_TEST)
        print(f"\n  Haus B cross-temporal diagnostic skipped "
              f"(not in gmm_eval_series: {missing})")

    # Save results.
    out_path = results_dir / 'detector_eval_results.json'
    # Load existing file if present (to accumulate steps).
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  detector_eval_results.json → {out_path}")

    # Summary table.
    print("\n  Summary:")
    print(f"  {'Detector':<30} {'Mean AUC':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for det, r in results.items():
        print(f"  {det:<30} {r.get('mean_auc', float('nan')):>10.4f}")

    return results


def tune_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """Optuna hyperparameter search. Optimises mean test ROC-AUC over spring series."""
    import optuna

    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    img_out = (img_path / 'contrastive' / 'optuna') if img_path else None
    if img_out:
        img_out.mkdir(parents=True, exist_ok=True)

    n_trials = int(cfg.get('optuna_n_trials', 200))
    n_startup = int(cfg.get('optuna_n_startup_trials', 20))
    warmup_steps = int(cfg.get('optuna_warmup_steps', 15))
    epochs = int(cfg.get('epochs', 80))
    patience = int(cfg.get('patience', 15))
    keep_top_n = int(cfg.get('keep_top_n_checkpoints', 3))
    space = cfg.get('optuna_search_space', {})
    encoder_type = cfg.get('encoder_type', 'autoencoder')

    encoder_arch = cfg.get('encoder_arch', 'cnn')

    print("=" * 60)
    print(f"CONTRASTIVE OPTUNA STUDY ({encoder_arch.upper()} / {encoder_type})")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Trials:        {n_trials}  (startup={n_startup} random, then TPE)")
    print(f"Pruner:        MedianPruner (warmup_steps={warmup_steps})")
    print(f"Epochs/trial:  {epochs} (patience={patience}) + pruning")
    print(f"Objective:     mean test ROC-AUC over spring series")
    print(f"Keep top-N:    {keep_top_n} checkpoints")
    print("=" * 60)

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    train_ids = splits_data['splits']['train']
    val_ids = splits_data['splits']['val']
    test_ids = splits_data['splits']['test']

    print("\n--- LOADING DATA (once, reused across all trials) ---")
    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)
    selected = _resolve_selected_features(cfg)
    data = _prepare_dataset_tensors(
        train_ids, val_ids, test_ids, prepared_base, metadata_by_id,
        selected,
        cfg.get('scaler', 'standard'),
        dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
        regress_target_map=regress_map,
    )

    contrastive_loss = str(cfg.get('contrastive_loss', 'supcon')).lower()
    use_supcon = contrastive_loss in ('supcon', 'both', 'supcon+rnc', 'rnc+supcon')
    use_rnc = contrastive_loss in ('rnc', 'both', 'supcon+rnc', 'rnc+supcon')
    use_deepsad = contrastive_loss == 'deepsad'

    # Per-patch file-name array — needed by the fair NoFlow calibration/test
    # split (_evaluate_on_test_fair requires file-level grouping). Built once,
    # reused across all trials, same construction as train_contrastive/eval_contrastive.
    test_files_list_tune: List[str] = []
    for seg_id in test_ids:
        meta = metadata_by_id.get(seg_id)
        if meta is None or 'patch_file' not in meta:
            continue
        patch_path = prepared_base / meta['series'] / meta['patch_file']
        if not patch_path.exists():
            continue
        n = meta.get('n_patches', 0)
        if n == 0:
            continue
        test_files_list_tune.extend([meta['file_name']] * n)
    test_files_arr_tune = np.array(test_files_list_tune) if test_files_list_tune else None

    # Track top-N checkpoints (by objective)
    top_checkpoints: List[Tuple[float, int, Dict]] = []  # (objective, trial_number, state_dict)
    checkpoints_dir = results_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: 'optuna.Trial') -> float:
        _is_transformer = (encoder_arch == 'transformer')
        hp = {
            'encoder_type': encoder_type,
            'encoder_arch': encoder_arch,
            'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
            'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
            'dropout': cfg.get('dropout', 0.3),
            'latent_dim': trial.suggest_categorical(
                'latent_dim', space.get('latent_dim', [8, 16, 32, 64, 128])),
            'projection_dim': trial.suggest_categorical(
                'projection_dim', space.get('projection_dim', [32, 64, 128])),
            'k_per_series': trial.suggest_categorical(
                'k_per_series', space.get('k_per_series', [4, 8, 16, 32])),
            'contrastive_loss': contrastive_loss,
            'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
            'lr': trial.suggest_float(
                'lr', *space.get('lr', [1e-4, 5e-3]), log=True),
        }
        # Transformer-specific HPs (only when encoder_arch == 'transformer')
        if _is_transformer:
            d_model = trial.suggest_categorical(
                'd_model', space.get('d_model', [64, 128, 256]))
            all_heads = space.get('num_heads', [2, 4, 8])
            num_heads_raw = trial.suggest_categorical('num_heads', all_heads)
            if d_model % num_heads_raw != 0:
                valid = sorted(h for h in all_heads if d_model % h == 0)
                num_heads = valid[-1] if valid else 1
            else:
                num_heads = num_heads_raw
            hp.update({
                'd_model': d_model,
                'num_heads': num_heads,
                'num_layers': trial.suggest_categorical(
                    'num_layers', space.get('num_layers', [2, 4, 6])),
                'ff_dim': trial.suggest_categorical(
                    'ff_dim', space.get('ff_dim', [128, 256, 512])),
                'transformer_dropout': trial.suggest_float(
                    'transformer_dropout',
                    *space.get('transformer_dropout', [0.05, 0.3])),
                'warmup_epochs': trial.suggest_categorical(
                    'warmup_epochs', space.get('warmup_epochs', [5, 10, 20])),
            })
        else:
            hp.update({
                'd_model': cfg.get('d_model', 128),
                'num_heads': cfg.get('num_heads', 4),
                'num_layers': cfg.get('num_layers', 4),
                'ff_dim': cfg.get('ff_dim', 256),
                'transformer_dropout': cfg.get('transformer_dropout', 0.1),
                # CNN default warmup is 0, but overridable via CLI (e.g. --warmup-epochs 10
                # is recommended when using DeepSAD to prevent best_epoch=0 pathology).
                'warmup_epochs': cfg.get('warmup_epochs', 0),
            })
        # Only tune hyperparams for the active loss; zero out unused ones.
        if use_supcon:
            hp['lambda_supcon'] = trial.suggest_float(
                'lambda_supcon', *space.get('lambda_supcon', [0.05, 5.0]), log=True)
            hp['supcon_temperature'] = trial.suggest_float(
                'supcon_temperature', *space.get('supcon_temperature', [0.03, 0.5]), log=True)
        else:
            hp['lambda_supcon'] = 0.0
            hp['supcon_temperature'] = 0.1  # unused placeholder
        if use_rnc:
            hp['lambda_rnc'] = trial.suggest_float(
                'lambda_rnc', *space.get('lambda_rnc', [0.05, 5.0]), log=True)
            hp['rnc_temperature'] = trial.suggest_float(
                'rnc_temperature', *space.get('rnc_temperature', [0.5, 8.0]), log=True)
        else:
            hp['lambda_rnc'] = 0.0
            hp['rnc_temperature'] = 2.0  # unused placeholder
        if use_deepsad:
            hp['lambda_deepsad'] = trial.suggest_float(
                'lambda_deepsad', *space.get('lambda_deepsad', [0.05, 5.0]), log=True)
            hp['deepsad_margin'] = trial.suggest_float(
                'deepsad_margin', *space.get('deepsad_margin', [0.3, 1.5]))
            hp['deepsad_repulsion_weight'] = cfg.get('deepsad_repulsion_weight', 1.0)
        else:
            hp['lambda_deepsad'] = 0.0
            hp['deepsad_margin'] = 1.0           # unused placeholder
            hp['deepsad_repulsion_weight'] = 1.0  # unused placeholder

        try:
            model, history, best_epoch = _run_single_training(
                data, hp, device, epochs=epochs, patience=patience,
                optuna_trial=trial, verbose=False,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  [trial {trial.number}] failed: {e}")
            raise optuna.TrialPruned()

        test_metrics, _, _ = _evaluate_on_test_fair(
            model, data, device,
            gmm_n_components=cfg.get('gmm_n_components', 1),
            gmm_covariance_type=cfg.get('gmm_covariance_type', 'full'),
            test_files=test_files_arr_tune,
            mode=cfg.get('eval_noflow_split_mode', 'fair'),
            calibration_file_ratio=cfg.get('eval_calibration_file_ratio', 0.5),
            min_calibration_patches=cfg.get('eval_min_calibration_patches', 500),
        )
        objective_value = test_metrics.get('overall', {}).get('mean_roc_auc', 0.0)

        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('n_params', sum(p.numel() for p in model.parameters()))
        for series_name, m in test_metrics.items():
            if series_name == 'overall':
                continue
            trial.set_user_attr(f'auc_{series_name}', m.get('roc_auc'))
            trial.set_user_attr(f'f1_{series_name}', m.get('f1'))

        # Maintain top-N in memory + disk
        entry = (objective_value, trial.number, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        top_checkpoints.append(entry)
        top_checkpoints.sort(key=lambda t: -t[0])  # higher = better
        while len(top_checkpoints) > keep_top_n:
            _, dropped_num, _ = top_checkpoints.pop()
            p = checkpoints_dir / f'trial_{dropped_num:04d}.pt'
            if p.exists():
                p.unlink()

        ckpt_path = checkpoints_dir / f'trial_{trial.number:04d}.pt'
        torch.save({
            'model_state_dict': entry[2],
            'hp': hp,
            'trial_number': trial.number,
            'objective': objective_value,
            'test_metrics': test_metrics,
            'input_height': data['input_height'],
            'input_width': data['input_width'],
            'n_in_channels': data['n_in_channels'],
        }, ckpt_path)

        return objective_value

    storage_path = results_dir / 'optuna_study.db'
    storage_url = f'sqlite:///{storage_path.as_posix()}'
    study_name = cfg.get('study_name', 'contrastive_default')
    study = optuna.create_study(
        direction='maximize',
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup, n_warmup_steps=warmup_steps,
        ),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 60}\nBEST TRIAL\n{'=' * 60}")
    print(f"  #{best.number}  objective={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:22s}: {v}")
    for k, v in best.user_attrs.items():
        print(f"  [attr] {k:20s}: {v}")

    # Persist best params + study summary. The `variant` block records which
    # loss recipe was actually tuned so a subsequent `--run train` / `--run loso`
    # reproduces it even if the JSON default drifts (see
    # `_apply_variant_from_best_params`).
    best_path = results_dir / 'optuna_best_params.json'
    with open(best_path, 'w') as f:
        json.dump({
            'best_trial': best.number,
            'best_objective': best.value,
            'params': best.params,
            'user_attrs': best.user_attrs,
            'variant': _extract_variant_from_cfg({
                **cfg,
                'encoder_type': encoder_type,
                'encoder_arch': encoder_arch,
                'contrastive_loss': contrastive_loss,
            }),
        }, f, indent=2)

    top10 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: -t.value,
    )[:10]
    top10_path = results_dir / 'optuna_top10.json'
    with open(top10_path, 'w') as f:
        json.dump([
            {'number': t.number, 'value': t.value, 'params': t.params}
            for t in top10
        ], f, indent=2)

    print(f"\n  Saved: {best_path}")
    print(f"  Saved: {top10_path}")
    print(f"  Checkpoints: {checkpoints_dir} (top-{keep_top_n})")

    # Optuna plots if plot enabled
    if plot_enabled and img_out is not None:
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice,
            )
            plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(img_out / 'optimization_history.png', dpi=150)
            plt.close('all')

            plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(img_out / 'param_importances.png', dpi=150)
            plt.close('all')

            plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(img_out / 'parallel_coordinate.png', dpi=150)
            plt.close('all')

            plot_slice(study)
            plt.tight_layout()
            plt.savefig(img_out / 'slice.png', dpi=150)
            plt.close('all')
            print(f"  Saved Optuna plots to: {img_out}")
        except Exception as e:
            print(f"  [WARN] Optuna plotting failed: {e}")

    print("\n" + "=" * 60)
    print("OPTUNA STUDY COMPLETE")
    print("=" * 60)
    return {'best': best.params, 'objective': best.value}


def loso_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Leave-One-Series-Out evaluation. Uses the best Optuna params if available,
    otherwise the static config values. For each held-out series, trains on the
    other N-1 series and evaluates via a single GLOBAL GMM fitted on that
    training pool's NoFlow latents only — no per-site recalibration on the
    held-out series. See `_evaluate_global_gmm` for the fit itself.
    """
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prefer tuned hyperparameters if available
    best_params_path = results_dir / 'optuna_best_params.json'
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best = json.load(f)
        tuned = best.get('params', {})
        print(f"Using tuned hyperparameters from {best_params_path}")
        _apply_variant_from_best_params(cfg, best)
    else:
        tuned = {}
        print("No Optuna best params found — using config defaults.")

    loso_series = cfg.get('loso_series', [
        '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
        '7.03.2026', '15.03.2026',
    ])

    # Optional filter: only run these as held-out folds; training pool stays full.
    loso_held_out = cfg.get('loso_held_out', None)

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    all_ids = _ids_for_series(splits_data, metadata_by_id, loso_series)

    encoder_type = cfg.get('encoder_type', 'autoencoder')
    encoder_arch = cfg.get('encoder_arch', 'cnn')

    def _hp_for_run() -> Dict:
        _is_transformer = (encoder_arch == 'transformer')
        return {
            'encoder_type': encoder_type,
            'encoder_arch': encoder_arch,
            'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
            'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
            'dropout': cfg.get('dropout', 0.3),
            'latent_dim': tuned.get('latent_dim', cfg.get('latent_dim', 32)),
            'projection_dim': tuned.get('projection_dim', cfg.get('projection_dim', 64)),
            'k_per_series': tuned.get('k_per_series', cfg.get('k_per_series', 16)),
            'contrastive_loss': cfg.get('contrastive_loss', 'supcon'),
            'lambda_supcon': tuned.get('lambda_supcon', cfg.get('lambda_supcon', 0.5)),
            'supcon_temperature': tuned.get('supcon_temperature', cfg.get('supcon_temperature', 0.1)),
            'lambda_rnc': tuned.get('lambda_rnc', cfg.get('lambda_rnc', 0.5)),
            'rnc_temperature': tuned.get('rnc_temperature', cfg.get('rnc_temperature', 2.0)),
            'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
            'lambda_deepsad': tuned.get('lambda_deepsad', cfg.get('lambda_deepsad', 1.0)),
            'deepsad_margin': tuned.get('deepsad_margin', cfg.get('deepsad_margin', 1.0)),
            'deepsad_repulsion_weight': cfg.get('deepsad_repulsion_weight', 1.0),
            'lr': tuned.get('lr', cfg.get('lr', 1e-3)),
            'd_model': tuned.get('d_model', cfg.get('d_model', 128)),
            'num_heads': tuned.get('num_heads', cfg.get('num_heads', 4)),
            'num_layers': tuned.get('num_layers', cfg.get('num_layers', 4)),
            'ff_dim': tuned.get('ff_dim', cfg.get('ff_dim', 256)),
            'transformer_dropout': tuned.get('transformer_dropout', cfg.get('transformer_dropout', 0.1)),
            'warmup_epochs': tuned.get('warmup_epochs', cfg.get('warmup_epochs', 0)),
        }

    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)
    selected = _resolve_selected_features(cfg)

    # Resume: load existing results so we can append without re-running done folds.
    loso_results: Dict[str, Dict] = {}
    majority_vote_results: Dict[str, Dict] = {}
    existing_results_path = results_dir / 'loso_results.json'
    existing_mv_path = results_dir / 'majority_vote_results.json'
    if existing_results_path.exists():
        with open(existing_results_path) as f:
            loso_results = json.load(f)
        loso_results.pop('_summary', None)
        print(f"Resuming: loaded {len(loso_results)} existing results from {existing_results_path}")
    if existing_mv_path.exists():
        with open(existing_mv_path) as f:
            majority_vote_results = json.load(f)

    folds_to_run = loso_held_out if loso_held_out is not None else loso_series
    for held_out in folds_to_run:
        if held_out in loso_results:
            print(f"\n[skip] {held_out} — already in results, use --loso-held-out to re-run")
            continue

        print("\n" + "=" * 60)
        print(f"LOSO: held-out = {held_out}")
        print("=" * 60)

        train_side_ids = [
            sid for sid in all_ids
            if metadata_by_id[sid]['series'] != held_out
        ]
        held_out_ids = [
            sid for sid in all_ids
            if metadata_by_id[sid]['series'] == held_out
        ]

        # Build per-patch file origin array using the same iteration order
        # as _load_prepared_data so indices align with data['test_patches'].
        test_files_list: List[str] = []
        for seg_id in held_out_ids:
            meta = metadata_by_id.get(seg_id)
            if meta is None or 'patch_file' not in meta:
                continue
            patch_path = prepared_base / meta['series'] / meta['patch_file']
            if not patch_path.exists():
                continue
            n = meta.get('n_patches', 0)
            if n == 0:
                continue
            test_files_list.extend([meta['file_name']] * n)
        test_files_arr = np.array(test_files_list)

        data = _prepare_dataset_tensors(
            train_ids=train_side_ids,
            val_ids=[],
            test_ids=held_out_ids,
            prepared_base=prepared_base,
            metadata_by_id=metadata_by_id,
            selected_features=selected,
            scaler_type=cfg.get('scaler', 'standard'),
            dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
            regress_target_map=regress_map,
        )

        hp = _hp_for_run()
        model, _history, _best_ep = _run_single_training(
            data, hp, device,
            epochs=cfg.get('epochs', 80),
            patience=cfg.get('patience', 15),
            verbose=False,
        )

        # ═══════════════════════════════════════════════════════════════
        # GLOBAL GMM (LOSO) — fit once on this fold's TRAINING POOL only
        # (the other N-1 series), never on the held-out series itself.
        # This is the deployment-realistic setting: no per-site
        # recalibration on `held_out` before scoring it. Replaces the
        # previous per-series call (`_evaluate_on_test`), which fit a
        # fresh GMM on the held-out series' own NoFlow latents.
        # ═══════════════════════════════════════════════════════════════
        metrics, mv_for_series = _evaluate_global_gmm(
            model, data, device,
            gmm_n_components=cfg.get('gmm_n_components', 1),
            gmm_covariance_type=cfg.get('gmm_covariance_type', 'full'),
            test_files=test_files_arr,
        )
        series_m = metrics.get(held_out, {})
        roc = series_m.get('roc_auc')
        f1 = series_m.get('f1')
        print(f"  ROC-AUC: {roc:.4f} | F1: {f1:.4f}" if roc is not None else "  (skipped)")
        loso_results[held_out] = series_m

        # Print majority voting per file
        if held_out in mv_for_series:
            print(f"\n  {held_out}")
            for fname, finfo in sorted(mv_for_series[held_out].items()):
                print(f"    {fname}  {finfo['true_label']}")
                print(f"      NoFlow:  {finfo['n_noflow']}")
                print(f"      Leakage: {finfo['n_leakage']}")
        majority_vote_results[held_out] = mv_for_series.get(held_out, {})

    # Aggregate — exclude NoFlow-only series (n_flow=0 → AUC=0.5 is undefined)
    aucs = [m['roc_auc'] for m in loso_results.values()
            if 'roc_auc' in m and m.get('n_flow', 0) > 0]
    f1s = [m['f1'] for m in loso_results.values()
           if 'f1' in m and m.get('n_flow', 0) > 0]
    loso_results['_summary'] = {
        'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
        'mean_f1': float(np.mean(f1s)) if f1s else None,
        'n_series': len(aucs),
    }

    out_path = results_dir / 'loso_results.json'
    with open(out_path, 'w') as f:
        json.dump(loso_results, f, indent=2)

    mv_path = results_dir / 'majority_vote_results.json'
    with open(mv_path, 'w') as f:
        json.dump(majority_vote_results, f, indent=2)

    print("\n" + "=" * 60)
    print("LOSO SUMMARY")
    print("=" * 60)
    for k, v in loso_results.items():
        print(f"  {k}: {v}")
    print(f"  Saved: {out_path}")
    print(f"  Saved majority vote: {mv_path}")

    return loso_results


def chronological_extrapolation_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Chronological (walk-forward) extrapolation study. Series are sorted by
    date via `chrono_series` (defaults to the same chronologically-ordered
    list as `loso_series`). For each split index k starting at
    `chrono_min_train` (default 4): train on series[0:k], test on the SINGLE
    next series series[k] — i.e. train-1..4/test-5, train-1..5/test-6, ...,
    train-1..(N-1)/test-N.

    This is deliberately NOT leave-one-out: unlike `loso_contrastive`, the
    training pool for split k never includes any series chronologically
    AFTER the held-out one. It answers "if you'd only ever seen the past,
    how well would you generalise to the very next new series?" — the
    genuine deployment question, since a real installation never has future
    data during training. GMM is fit once per split on that split's
    training pool only (`_evaluate_global_gmm`), scored on the next series
    with no recalibration — consistent with the LOSO Global-GMM setting.
    """
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = results_dir / 'optuna_best_params.json'
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best = json.load(f)
        tuned = best.get('params', {})
        print(f"Using tuned hyperparameters from {best_params_path}")
        _apply_variant_from_best_params(cfg, best)
    else:
        tuned = {}
        print("No Optuna best params found — using config defaults.")

    # Reuses the loso_series default (already chronologically ordered) unless
    # a dedicated chrono_series list is set — avoids maintaining two
    # near-identical series lists in encoder.json.
    chrono_series = cfg.get('chrono_series') or cfg.get('loso_series', [
        '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
        '7.03.2026', '15.03.2026',
    ])
    min_train = cfg.get('chrono_min_train', 4)
    # Optional filter: only run these test-series splits; e.g. to resume or
    # re-run a subset. Mirrors --loso-held-out.
    chrono_test_series = cfg.get('chrono_test_series', None)

    if min_train >= len(chrono_series):
        raise ValueError(
            f"chrono_min_train ({min_train}) must be less than "
            f"len(chrono_series) ({len(chrono_series)}) — no splits possible."
        )

    splits = [
        (chrono_series[:k], chrono_series[k])
        for k in range(min_train, len(chrono_series))
    ]
    if chrono_test_series is not None:
        splits = [(tr, te) for tr, te in splits if te in chrono_test_series]

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    encoder_type = cfg.get('encoder_type', 'autoencoder')
    encoder_arch = cfg.get('encoder_arch', 'cnn')

    def _hp_for_run() -> Dict:
        _is_transformer = (encoder_arch == 'transformer')
        return {
            'encoder_type': encoder_type,
            'encoder_arch': encoder_arch,
            'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
            'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
            'dropout': cfg.get('dropout', 0.3),
            'latent_dim': tuned.get('latent_dim', cfg.get('latent_dim', 32)),
            'projection_dim': tuned.get('projection_dim', cfg.get('projection_dim', 64)),
            'k_per_series': tuned.get('k_per_series', cfg.get('k_per_series', 16)),
            'contrastive_loss': cfg.get('contrastive_loss', 'supcon'),
            'lambda_supcon': tuned.get('lambda_supcon', cfg.get('lambda_supcon', 0.5)),
            'supcon_temperature': tuned.get('supcon_temperature', cfg.get('supcon_temperature', 0.1)),
            'lambda_rnc': tuned.get('lambda_rnc', cfg.get('lambda_rnc', 0.5)),
            'rnc_temperature': tuned.get('rnc_temperature', cfg.get('rnc_temperature', 2.0)),
            'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
            'lambda_deepsad': tuned.get('lambda_deepsad', cfg.get('lambda_deepsad', 1.0)),
            'deepsad_margin': tuned.get('deepsad_margin', cfg.get('deepsad_margin', 1.0)),
            'deepsad_repulsion_weight': cfg.get('deepsad_repulsion_weight', 1.0),
            'lr': tuned.get('lr', cfg.get('lr', 1e-3)),
            'd_model': tuned.get('d_model', cfg.get('d_model', 128)),
            'num_heads': tuned.get('num_heads', cfg.get('num_heads', 4)),
            'num_layers': tuned.get('num_layers', cfg.get('num_layers', 4)),
            'ff_dim': tuned.get('ff_dim', cfg.get('ff_dim', 256)),
            'transformer_dropout': tuned.get('transformer_dropout', cfg.get('transformer_dropout', 0.1)),
            'warmup_epochs': tuned.get('warmup_epochs', cfg.get('warmup_epochs', 0)),
        }

    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)
    selected = _resolve_selected_features(cfg)

    # Resume: load existing results so we can append without re-running done splits.
    chrono_results: Dict[str, Dict] = {}
    majority_vote_results: Dict[str, Dict] = {}
    existing_results_path = results_dir / 'chrono_extrap_results.json'
    existing_mv_path = results_dir / 'chrono_extrap_majority_vote_results.json'
    if existing_results_path.exists():
        with open(existing_results_path) as f:
            chrono_results = json.load(f)
        chrono_results.pop('_summary', None)
        print(f"Resuming: loaded {len(chrono_results)} existing results from {existing_results_path}")
    if existing_mv_path.exists():
        with open(existing_mv_path) as f:
            majority_vote_results = json.load(f)

    for train_series, test_series in splits:
        split_key = f"train1-{len(train_series)}_test_{test_series}"
        if split_key in chrono_results:
            print(f"\n[skip] {split_key} — already in results")
            continue

        print("\n" + "=" * 60)
        print(f"CHRONOLOGICAL EXTRAPOLATION: train on {len(train_series)} series "
              f"({train_series[0]} .. {train_series[-1]}) -> test {test_series}")
        print("=" * 60)

        all_ids = _ids_for_series(splits_data, metadata_by_id, train_series + [test_series])
        train_side_ids = [
            sid for sid in all_ids
            if metadata_by_id[sid]['series'] in train_series
        ]
        test_ids = [
            sid for sid in all_ids
            if metadata_by_id[sid]['series'] == test_series
        ]

        test_files_list: List[str] = []
        for seg_id in test_ids:
            meta = metadata_by_id.get(seg_id)
            if meta is None or 'patch_file' not in meta:
                continue
            patch_path = prepared_base / meta['series'] / meta['patch_file']
            if not patch_path.exists():
                continue
            n = meta.get('n_patches', 0)
            if n == 0:
                continue
            test_files_list.extend([meta['file_name']] * n)
        test_files_arr = np.array(test_files_list)

        data = _prepare_dataset_tensors(
            train_ids=train_side_ids,
            val_ids=[],
            test_ids=test_ids,
            prepared_base=prepared_base,
            metadata_by_id=metadata_by_id,
            selected_features=selected,
            scaler_type=cfg.get('scaler', 'standard'),
            dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
            regress_target_map=regress_map,
        )

        hp = _hp_for_run()
        model, _history, _best_ep = _run_single_training(
            data, hp, device,
            epochs=cfg.get('epochs', 80),
            patience=cfg.get('patience', 15),
            verbose=False,
        )

        # Global GMM fit on this split's training pool only (series[0:k]) —
        # the test series is chronologically AFTER every training series, so
        # this is genuine forward extrapolation, not interpolation.
        metrics, mv_for_series = _evaluate_global_gmm(
            model, data, device,
            gmm_n_components=cfg.get('gmm_n_components', 1),
            gmm_covariance_type=cfg.get('gmm_covariance_type', 'full'),
            test_files=test_files_arr,
        )
        series_m = metrics.get(test_series, {})
        series_m['n_train_series'] = len(train_series)
        series_m['train_series'] = train_series
        series_m['test_series'] = test_series
        roc = series_m.get('roc_auc')
        f1 = series_m.get('f1')
        print(f"  ROC-AUC: {roc:.4f} | F1: {f1:.4f}" if roc is not None else "  (skipped — no/single-class data)")
        chrono_results[split_key] = series_m

        if test_series in mv_for_series:
            print(f"\n  {test_series}")
            for fname, finfo in sorted(mv_for_series[test_series].items()):
                print(f"    {fname}  {finfo['true_label']}")
                print(f"      NoFlow:  {finfo['n_noflow']}")
                print(f"      Leakage: {finfo['n_leakage']}")
        majority_vote_results[split_key] = mv_for_series.get(test_series, {})

        # Save after every split — a multi-hour run shouldn't lose earlier
        # splits if a later one crashes or the job gets killed.
        with open(existing_results_path, 'w') as f:
            json.dump(chrono_results, f, indent=2)
        with open(existing_mv_path, 'w') as f:
            json.dump(majority_vote_results, f, indent=2)

    aucs = [m['roc_auc'] for m in chrono_results.values()
            if 'roc_auc' in m and m.get('n_flow', 0) > 0]
    f1s = [m['f1'] for m in chrono_results.values()
           if 'f1' in m and m.get('n_flow', 0) > 0]
    chrono_results['_summary'] = {
        'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
        'mean_f1': float(np.mean(f1s)) if f1s else None,
        'n_splits': len(aucs),
        'chrono_series': chrono_series,
        'chrono_min_train': min_train,
    }

    with open(existing_results_path, 'w') as f:
        json.dump(chrono_results, f, indent=2)
    with open(existing_mv_path, 'w') as f:
        json.dump(majority_vote_results, f, indent=2)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL EXTRAPOLATION SUMMARY")
    print("=" * 60)
    for k, v in chrono_results.items():
        print(f"  {k}: {v}")
    print(f"  Saved: {existing_results_path}")
    print(f"  Saved majority vote: {existing_mv_path}")

    return chrono_results


def chronological_extrapolation_temperature_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Same walk-forward splits as `chronological_extrapolation_contrastive`
    (train on series[0:k], test on the single next series series[k]), but
    each split's trained model is scored THREE ways instead of one: plain
    global GMM ("none"), temperature-conditioned on temp_pipe only
    ("pipe"), and jointly on temp_pipe + temp_amb ("pipe_amb"). One
    training per split (not three) — the three scores reuse the same
    trained encoder, mirroring how step 8's `--run eval` reused a single
    checkpoint across temperature settings.

    Writes three parallel result files (+ matching majority-vote files):
    chrono_temp_none_results.json, chrono_temp_pipe_results.json,
    chrono_temp_pipe_amb_results.json.
    """
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = results_dir / 'optuna_best_params.json'
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best = json.load(f)
        tuned = best.get('params', {})
        print(f"Using tuned hyperparameters from {best_params_path}")
        _apply_variant_from_best_params(cfg, best)
    else:
        tuned = {}
        print("No Optuna best params found — using config defaults.")

    chrono_series = cfg.get('chrono_series') or cfg.get('loso_series', [
        '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
        '7.03.2026', '15.03.2026',
    ])
    min_train = cfg.get('chrono_min_train', 4)
    chrono_test_series = cfg.get('chrono_test_series', None)
    temp_target = cfg.get('gmm_temp_target', 'temp_pipe')
    temp_target = temp_target if isinstance(temp_target, str) else temp_target[0]

    if min_train >= len(chrono_series):
        raise ValueError(
            f"chrono_min_train ({min_train}) must be less than "
            f"len(chrono_series) ({len(chrono_series)}) — no splits possible."
        )

    splits = [
        (chrono_series[:k], chrono_series[k])
        for k in range(min_train, len(chrono_series))
    ]
    if chrono_test_series is not None:
        splits = [(tr, te) for tr, te in splits if te in chrono_test_series]

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    encoder_type = cfg.get('encoder_type', 'autoencoder')
    encoder_arch = cfg.get('encoder_arch', 'cnn')

    def _hp_for_run() -> Dict:
        _is_transformer = (encoder_arch == 'transformer')
        return {
            'encoder_type': encoder_type,
            'encoder_arch': encoder_arch,
            'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
            'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
            'dropout': cfg.get('dropout', 0.3),
            'latent_dim': tuned.get('latent_dim', cfg.get('latent_dim', 32)),
            'projection_dim': tuned.get('projection_dim', cfg.get('projection_dim', 64)),
            'k_per_series': tuned.get('k_per_series', cfg.get('k_per_series', 16)),
            'contrastive_loss': cfg.get('contrastive_loss', 'supcon'),
            'lambda_supcon': tuned.get('lambda_supcon', cfg.get('lambda_supcon', 0.5)),
            'supcon_temperature': tuned.get('supcon_temperature', cfg.get('supcon_temperature', 0.1)),
            'lambda_rnc': tuned.get('lambda_rnc', cfg.get('lambda_rnc', 0.5)),
            'rnc_temperature': tuned.get('rnc_temperature', cfg.get('rnc_temperature', 2.0)),
            'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
            'lambda_deepsad': tuned.get('lambda_deepsad', cfg.get('lambda_deepsad', 1.0)),
            'deepsad_margin': tuned.get('deepsad_margin', cfg.get('deepsad_margin', 1.0)),
            'deepsad_repulsion_weight': cfg.get('deepsad_repulsion_weight', 1.0),
            'lr': tuned.get('lr', cfg.get('lr', 1e-3)),
            'd_model': tuned.get('d_model', cfg.get('d_model', 128)),
            'num_heads': tuned.get('num_heads', cfg.get('num_heads', 4)),
            'num_layers': tuned.get('num_layers', cfg.get('num_layers', 4)),
            'ff_dim': tuned.get('ff_dim', cfg.get('ff_dim', 256)),
            'transformer_dropout': tuned.get('transformer_dropout', cfg.get('transformer_dropout', 0.1)),
            'warmup_epochs': tuned.get('warmup_epochs', cfg.get('warmup_epochs', 0)),
        }

    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)
    selected = _resolve_selected_features(cfg)

    SETTINGS = ['none', 'pipe', 'pipe_amb']
    results: Dict[str, Dict[str, Dict]] = {}
    mv_results: Dict[str, Dict[str, Dict]] = {}
    paths: Dict[str, Path] = {}
    for setting in SETTINGS:
        results[setting] = {}
        mv_results[setting] = {}
        paths[setting] = results_dir / f'chrono_temp_{setting}_results.json'
        mv_path_s = results_dir / f'chrono_temp_{setting}_majority_vote_results.json'
        if paths[setting].exists():
            with open(paths[setting]) as f:
                results[setting] = json.load(f)
            results[setting].pop('_summary', None)
        if mv_path_s.exists():
            with open(mv_path_s) as f:
                mv_results[setting] = json.load(f)
    if results['none']:
        print(f"Resuming: loaded {len(results['none'])} existing splits from {paths['none']}")

    for train_series, test_series in splits:
        split_key = f"train1-{len(train_series)}_test_{test_series}"
        if split_key in results['none']:
            print(f"\n[skip] {split_key} — already in results")
            continue

        print("\n" + "=" * 60)
        print(f"CHRONOLOGICAL EXTRAPOLATION x TEMPERATURE: train on {len(train_series)} series "
              f"({train_series[0]} .. {train_series[-1]}) -> test {test_series}")
        print("=" * 60)

        all_ids = _ids_for_series(splits_data, metadata_by_id, train_series + [test_series])
        train_side_ids = [sid for sid in all_ids if metadata_by_id[sid]['series'] in train_series]
        test_ids = [sid for sid in all_ids if metadata_by_id[sid]['series'] == test_series]

        test_files_list: List[str] = []
        for seg_id in test_ids:
            meta = metadata_by_id.get(seg_id)
            if meta is None or 'patch_file' not in meta:
                continue
            patch_path = prepared_base / meta['series'] / meta['patch_file']
            if not patch_path.exists():
                continue
            n = meta.get('n_patches', 0)
            if n == 0:
                continue
            test_files_list.extend([meta['file_name']] * n)
        test_files_arr = np.array(test_files_list)

        data = _prepare_dataset_tensors(
            train_ids=train_side_ids,
            val_ids=[],
            test_ids=test_ids,
            prepared_base=prepared_base,
            metadata_by_id=metadata_by_id,
            selected_features=selected,
            scaler_type=cfg.get('scaler', 'standard'),
            dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
            regress_target_map=regress_map,
        )

        hp = _hp_for_run()
        model, _history, _best_ep = _run_single_training(
            data, hp, device,
            epochs=cfg.get('epochs', 80),
            patience=cfg.get('patience', 15),
            verbose=False,
        )

        # --- Setting 1: no temperature (plain global GMM) ---
        metrics_none, mv_none = _evaluate_global_gmm(
            model, data, device,
            gmm_n_components=cfg.get('gmm_n_components', 1),
            gmm_covariance_type=cfg.get('gmm_covariance_type', 'full'),
            test_files=test_files_arr,
        )
        series_m = metrics_none.get(test_series, {})
        series_m.update({'n_train_series': len(train_series), 'train_series': train_series, 'test_series': test_series})
        results['none'][split_key] = series_m
        mv_results['none'][split_key] = mv_none.get(test_series, {})
        roc = series_m.get('roc_auc')
        print(f"  [none]     ROC-AUC: {roc:.4f}" if roc is not None else "  [none]     (skipped)")

        # --- Settings 2+3 need per-series temperature maps for this split's series union ---
        unique_series = sorted(set(list(data['train_series']) + list(data['test_series'])))
        temp_map_pipe = _compute_per_series_regress_target(data_path, unique_series, temp_target)
        temp_map_amb = _compute_per_series_regress_target(data_path, unique_series, 'temp_amb')

        metrics_pipe, mv_pipe = _evaluate_global_gmm_conditional(
            model, data, device,
            gmm_n_components=cfg.get('gmm_n_components', 1),
            gmm_covariance_type=cfg.get('gmm_covariance_type', 'full'),
            temp_maps={temp_target: temp_map_pipe},
            temp_target_names=[temp_target],
            test_files=test_files_arr,
        )
        series_m = metrics_pipe.get(test_series, {})
        series_m.update({'n_train_series': len(train_series), 'train_series': train_series, 'test_series': test_series})
        results['pipe'][split_key] = series_m
        mv_results['pipe'][split_key] = mv_pipe.get(test_series, {})
        roc = series_m.get('roc_auc')
        print(f"  [pipe]     ROC-AUC: {roc:.4f}" if roc is not None else "  [pipe]     (skipped)")

        metrics_amb, mv_amb = _evaluate_global_gmm_conditional(
            model, data, device,
            gmm_n_components=cfg.get('gmm_n_components', 1),
            gmm_covariance_type=cfg.get('gmm_covariance_type', 'full'),
            temp_maps={temp_target: temp_map_pipe, 'temp_amb': temp_map_amb},
            temp_target_names=[temp_target, 'temp_amb'],
            test_files=test_files_arr,
        )
        series_m = metrics_amb.get(test_series, {})
        series_m.update({'n_train_series': len(train_series), 'train_series': train_series, 'test_series': test_series})
        results['pipe_amb'][split_key] = series_m
        mv_results['pipe_amb'][split_key] = mv_amb.get(test_series, {})
        roc = series_m.get('roc_auc')
        print(f"  [pipe_amb] ROC-AUC: {roc:.4f}" if roc is not None else "  [pipe_amb] (skipped)")

        # Save after every split, all three settings — a multi-hour run
        # shouldn't lose earlier splits if a later one crashes.
        for setting in SETTINGS:
            with open(paths[setting], 'w') as f:
                json.dump(results[setting], f, indent=2)
            mv_path_s = results_dir / f'chrono_temp_{setting}_majority_vote_results.json'
            with open(mv_path_s, 'w') as f:
                json.dump(mv_results[setting], f, indent=2)

    for setting in SETTINGS:
        aucs = [m['roc_auc'] for m in results[setting].values()
                if 'roc_auc' in m and m.get('n_flow', 0) > 0]
        f1s = [m['f1'] for m in results[setting].values()
               if 'f1' in m and m.get('n_flow', 0) > 0]
        results[setting]['_summary'] = {
            'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
            'mean_f1': float(np.mean(f1s)) if f1s else None,
            'n_splits': len(aucs),
            'setting': setting,
            'chrono_series': chrono_series,
            'chrono_min_train': min_train,
        }
        with open(paths[setting], 'w') as f:
            json.dump(results[setting], f, indent=2)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL EXTRAPOLATION x TEMPERATURE SUMMARY")
    print("=" * 60)
    for setting in SETTINGS:
        print(f"  [{setting}] {results[setting]['_summary']}")
        print(f"    Saved: {paths[setting]}")

    return results


def _pooled_metrics_from_scores(
    labels: np.ndarray, scores: np.ndarray, test_series_arr: np.ndarray,
) -> Dict:
    """
    Pooled patch-level classification metrics from flat label/score arrays
    spanning MULTIPLE test series treated as one combined test set — the
    single-row-per-experiment metric the pooled chronological experiment
    needs, as opposed to per-series-then-averaged (what every other GMM
    eval function in this module computes).

    ROC-AUC is computed on the full pooled score/label arrays — threshold-
    independent, unaffected by anything below.

    Classification threshold is calibrated PER SERIES (each series' own
    mean + 3*sigma from ITS OWN NoFlow scores under the shared model),
    matching `_evaluate_global_gmm`'s established convention elsewhere in
    this module. Predictions are pooled AFTER per-series thresholding, not
    before. This replaces an earlier version that computed one threshold
    from the ENTIRE pooled NoFlow population across all test-range series —
    confirmed empirically to fail badly (e.g. AUC=0.855 but F1=0.001 on one
    split, 2026-07-13): pooling NoFlow scores from series with different
    baseline levels inflates the pooled sigma enormously, pushing
    mean+3*sigma so high that almost nothing gets flagged, even when the
    ranking itself (AUC) is good. A series with no NoFlow patches in this
    test range can't calibrate locally; its patches are left predicted as
    NoFlow (0), matching the "unknown -> assume no leakage" deployment
    default used elsewhere.
    """
    auc = _auc_safe(labels, scores)

    preds = np.zeros(len(labels), dtype=int)
    for s in sorted(set(test_series_arr.tolist())):
        mask = test_series_arr == s
        noflow_mask_s = mask & (labels == 0)
        if noflow_mask_s.sum() == 0:
            continue
        mu = float(scores[noflow_mask_s].mean())
        sigma = float(scores[noflow_mask_s].std() + 1e-12)
        thr = mu + 3.0 * sigma
        preds[mask] = (scores[mask] > thr).astype(int)

    n_correct = int((preds == labels).sum())
    n_wrong = len(labels) - n_correct
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'roc_auc': auc,
        'f1': f1,
        'n_noflow': int((labels == 0).sum()),
        'n_flow': int((labels == 1).sum()),
        'accuracy': n_correct / len(labels) if len(labels) else 0.0,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }


def _pooled_metrics_fixed_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float,
) -> Dict:
    """
    Strict counterpart to `_pooled_metrics_from_scores`: ONE threshold,
    estimated from training-NoFlow scores only (never from any future test
    series), applied unchanged to every pooled test patch regardless of
    which series it came from. Models the deployment scenario where NO
    calibration period is assumed to exist at the new site — a harder,
    more pessimistic bound than the per-series-calibrated protocol above.
    ROC-AUC is threshold-independent and will match the calibrated-protocol
    value exactly for the same split; only F1/accuracy/confusion matrix can
    differ.
    """
    auc = _auc_safe(labels, scores)
    preds = (scores > threshold).astype(int)

    n_correct = int((preds == labels).sum())
    n_wrong = len(labels) - n_correct
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'roc_auc': auc,
        'f1': f1,
        'threshold': float(threshold),
        'n_noflow': int((labels == 0).sum()),
        'n_flow': int((labels == 1).sum()),
        'accuracy': n_correct / len(labels) if len(labels) else 0.0,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }


def chronological_extrapolation_pooled_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Matches the supervisor's sketch exactly: for split k, train on
    chrono_series[0:k], test on ALL remaining series chrono_series[k:]
    POOLED together into one flat test set (train 1-4/test 5-N, train
    1-5/test 6-N, ..., train 1-(N-1)/test N). Distinct from
    `chronological_extrapolation_contrastive` and `..._temperature_...`
    (both test one single next series per split) — this tests the ENTIRE
    remaining future as one pooled set each time, which is what "Train End
    | Test Range | Accuracy | F1 | ROC-AUC" (one row per experiment)
    requires: pooled ROC-AUC/F1/accuracy/confusion-matrix over the union of
    all test-range patches, not an average of per-series numbers.

    One training per split, scored 2 ways on the same model — plain global
    GMM ("none") and temp_pipe-conditioned global GMM ("pipe") — both
    computed pooled. "if available" (pipe+ambient) was requested but ambient
    conditioning showed zero effect beyond pipe alone in every prior test on
    this dataset (Eval C, step-8 interpolation-temperature) — pipe-only is
    the setting actually asked for by name here ("plus pipe temperature").
    """
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = results_dir / 'optuna_best_params.json'
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best = json.load(f)
        tuned = best.get('params', {})
        print(f"Using tuned hyperparameters from {best_params_path}")
        _apply_variant_from_best_params(cfg, best)
    else:
        tuned = {}
        print("No Optuna best params found — using config defaults.")

    # chrono_series (NOT loso_series) — see _comment_chrono in encoder.json
    # for why: loso_series has ha-6.06/ha-2.06 swapped, harmless for LOSO
    # (order-independent) but wrong for any experiment that depends on
    # chronological order, like this one.
    chrono_series = cfg.get('chrono_series') or cfg.get('loso_series', [
        '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
        '7.03.2026', '15.03.2026',
    ])
    min_train = cfg.get('chrono_min_train', 4)
    # Filter by the FIRST series of the test range (i.e. which split to run).
    chrono_test_series = cfg.get('chrono_test_series', None)
    temp_target = cfg.get('gmm_temp_target', 'temp_pipe')
    temp_target = temp_target if isinstance(temp_target, str) else temp_target[0]

    if min_train >= len(chrono_series) - 1:
        raise ValueError(
            f"chrono_min_train ({min_train}) leaves no room for a non-empty "
            f"test range with {len(chrono_series)} series."
        )

    splits = [
        (chrono_series[:k], chrono_series[k:])
        for k in range(min_train, len(chrono_series))
    ]
    if chrono_test_series is not None:
        splits = [(tr, te) for tr, te in splits if te[0] in chrono_test_series]

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    encoder_type = cfg.get('encoder_type', 'autoencoder')
    encoder_arch = cfg.get('encoder_arch', 'cnn')

    def _hp_for_run() -> Dict:
        _is_transformer = (encoder_arch == 'transformer')
        return {
            'encoder_type': encoder_type,
            'encoder_arch': encoder_arch,
            'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
            'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
            'dropout': cfg.get('dropout', 0.3),
            'latent_dim': tuned.get('latent_dim', cfg.get('latent_dim', 32)),
            'projection_dim': tuned.get('projection_dim', cfg.get('projection_dim', 64)),
            'k_per_series': tuned.get('k_per_series', cfg.get('k_per_series', 16)),
            'contrastive_loss': cfg.get('contrastive_loss', 'supcon'),
            'lambda_supcon': tuned.get('lambda_supcon', cfg.get('lambda_supcon', 0.5)),
            'supcon_temperature': tuned.get('supcon_temperature', cfg.get('supcon_temperature', 0.1)),
            'lambda_rnc': tuned.get('lambda_rnc', cfg.get('lambda_rnc', 0.5)),
            'rnc_temperature': tuned.get('rnc_temperature', cfg.get('rnc_temperature', 2.0)),
            'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
            'lambda_deepsad': tuned.get('lambda_deepsad', cfg.get('lambda_deepsad', 1.0)),
            'deepsad_margin': tuned.get('deepsad_margin', cfg.get('deepsad_margin', 1.0)),
            'deepsad_repulsion_weight': cfg.get('deepsad_repulsion_weight', 1.0),
            'lr': tuned.get('lr', cfg.get('lr', 1e-3)),
            'd_model': tuned.get('d_model', cfg.get('d_model', 128)),
            'num_heads': tuned.get('num_heads', cfg.get('num_heads', 4)),
            'num_layers': tuned.get('num_layers', cfg.get('num_layers', 4)),
            'ff_dim': tuned.get('ff_dim', cfg.get('ff_dim', 256)),
            'transformer_dropout': tuned.get('transformer_dropout', cfg.get('transformer_dropout', 0.1)),
            'warmup_epochs': tuned.get('warmup_epochs', cfg.get('warmup_epochs', 0)),
        }

    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)
    selected = _resolve_selected_features(cfg)

    SETTINGS = ['none', 'pipe']
    results: Dict[str, Dict[str, Dict]] = {s: {} for s in SETTINGS}
    paths: Dict[str, Path] = {s: results_dir / f'chrono_pooled_{s}_results.json' for s in SETTINGS}
    for s in SETTINGS:
        if paths[s].exists():
            with open(paths[s]) as f:
                results[s] = json.load(f)
            results[s].pop('_summary', None)
    if results['none']:
        print(f"Resuming: loaded {len(results['none'])} existing splits from {paths['none']}")

    for train_series, test_series_list in splits:
        split_key = f"train1-{len(train_series)}_test{len(train_series) + 1}-{len(chrono_series)}"
        if split_key in results['none']:
            print(f"\n[skip] {split_key} — already in results")
            continue

        print("\n" + "=" * 60)
        print(f"CHRONOLOGICAL EXTRAPOLATION (POOLED): train on {len(train_series)} series "
              f"({train_series[0]} .. {train_series[-1]}) -> test {len(test_series_list)} series "
              f"({test_series_list[0]} .. {test_series_list[-1]})")
        print("=" * 60)

        meta_common = {
            'n_train_series': len(train_series),
            'train_series': train_series,
            'n_test_series': len(test_series_list),
            'test_series_range': test_series_list,
        }

        all_ids = _ids_for_series(splits_data, metadata_by_id, train_series + test_series_list)
        train_side_ids = [sid for sid in all_ids if metadata_by_id[sid]['series'] in train_series]
        test_ids = [sid for sid in all_ids if metadata_by_id[sid]['series'] in test_series_list]

        data = _prepare_dataset_tensors(
            train_ids=train_side_ids,
            val_ids=[],
            test_ids=test_ids,
            prepared_base=prepared_base,
            metadata_by_id=metadata_by_id,
            selected_features=selected,
            scaler_type=cfg.get('scaler', 'standard'),
            dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
            regress_target_map=regress_map,
        )

        hp = _hp_for_run()
        model, _history, _best_ep = _run_single_training(
            data, hp, device,
            epochs=cfg.get('epochs', 80),
            patience=cfg.get('patience', 15),
            verbose=False,
        )

        gmm_n = cfg.get('gmm_n_components', 1)
        gmm_cov = cfg.get('gmm_covariance_type', 'full')

        if len(data['test_patches']) == 0 or len(data['train_patches']) == 0:
            print("  (no data)")
            for s in SETTINGS:
                results[s][split_key] = dict(meta_common)
        else:
            train_labels = np.asarray(data['train_labels'])
            train_noflow_mask = train_labels == 0
            train_latents_all = _extract_latents(model, data['train_patches'], device)
            train_noflow_lat = train_latents_all[train_noflow_mask]

            test_latents = _extract_latents(model, data['test_patches'], device)
            test_labels = np.asarray(data['test_labels'])
            test_series_arr = np.array(data['test_series'])

            if train_noflow_mask.sum() < max(gmm_n * 2, 4):
                print("  (not enough train NoFlow patches)")
                for s in SETTINGS:
                    results[s][split_key] = dict(meta_common)
            else:
                # --- Setting "none": plain global GMM, pooled over the test range ---
                try:
                    gmm_none = GaussianMixture(
                        n_components=gmm_n, covariance_type=gmm_cov,
                        reg_covar=1e-3, random_state=42,
                    )
                    gmm_none.fit(train_noflow_lat)
                    scores_none = -gmm_none.score_samples(test_latents)
                    m_none = _pooled_metrics_from_scores(test_labels, scores_none, test_series_arr)
                    m_none.update(meta_common)
                    m_none['gmm_fit_n_noflow'] = int(train_noflow_mask.sum())
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  [none] fit failed: {type(e).__name__}: {e}")
                    m_none = dict(meta_common)
                results['none'][split_key] = m_none
                roc = m_none.get('roc_auc')
                print(f"  [none] pooled ROC-AUC: {roc:.4f}" if roc is not None else "  [none] (skipped)")

                # --- Setting "pipe": temp_pipe-conditioned global GMM, pooled ---
                unique_series = sorted(set(list(data['train_series']) + list(data['test_series'])))
                temp_map_pipe = _compute_per_series_regress_target(data_path, unique_series, temp_target)
                train_series_arr = np.array(data['train_series'])
                train_temps = np.array(
                    [temp_map_pipe.get(s, np.nan) for s in train_series_arr], dtype=np.float64,
                )
                valid_train = ~np.isnan(train_temps)
                fit_mask = train_noflow_mask & valid_train
                n_unique_T = len(np.unique(train_temps[fit_mask])) if fit_mask.sum() else 0

                if fit_mask.sum() < max(gmm_n * 2, 4) or n_unique_T < 2:
                    print("  [pipe] insufficient training temperature data, skipping")
                    results['pipe'][split_key] = dict(meta_common)
                else:
                    T_fit = train_temps[fit_mask].reshape(-1, 1)
                    z_fit = train_latents_all[fit_mask]
                    A = np.hstack([T_fit, np.ones_like(T_fit)])
                    coef, *_ = np.linalg.lstsq(A, z_fit, rcond=None)
                    slope, intercept = coef[0], coef[1]
                    residual_train = z_fit - (T_fit @ slope[None, :] + intercept[None, :])
                    try:
                        gmm_pipe = GaussianMixture(
                            n_components=gmm_n, covariance_type=gmm_cov,
                            reg_covar=1e-3, random_state=42,
                        )
                        gmm_pipe.fit(residual_train)
                        # Each test patch scored against ITS OWN series'
                        # temperature-predicted centroid, then pooled.
                        scores_pipe = np.full(len(test_labels), np.nan)
                        for s in sorted(set(test_series_arr)):
                            mask = test_series_arr == s
                            T_s = temp_map_pipe.get(s, np.nan)
                            if np.isnan(T_s):
                                continue
                            z_pred_s = T_s * slope + intercept
                            residual_s = test_latents[mask] - z_pred_s[None, :]
                            scores_pipe[mask] = -gmm_pipe.score_samples(residual_s)
                        valid_test = ~np.isnan(scores_pipe)
                        if valid_test.sum() == 0 or len(set(test_labels[valid_test].tolist())) < 2:
                            print("  [pipe] no valid pooled test scores, skipping")
                            results['pipe'][split_key] = dict(meta_common)
                        else:
                            m_pipe = _pooled_metrics_from_scores(
                                test_labels[valid_test], scores_pipe[valid_test],
                                test_series_arr[valid_test],
                            )
                            m_pipe.update(meta_common)
                            m_pipe['gmm_fit_n_noflow'] = int(fit_mask.sum())
                            m_pipe['temp_target'] = temp_target
                            results['pipe'][split_key] = m_pipe
                            roc = m_pipe.get('roc_auc')
                            print(f"  [pipe] pooled ROC-AUC: {roc:.4f}" if roc is not None else "  [pipe] (skipped)")
                    except (np.linalg.LinAlgError, ValueError) as e:
                        print(f"  [pipe] fit failed: {type(e).__name__}: {e}")
                        results['pipe'][split_key] = dict(meta_common)

        # Save after every split, both settings — a multi-hour run
        # shouldn't lose earlier splits if a later one crashes.
        for s in SETTINGS:
            with open(paths[s], 'w') as f:
                json.dump(results[s], f, indent=2)

    for s in SETTINGS:
        aucs = [m['roc_auc'] for m in results[s].values()
                if 'roc_auc' in m and m.get('n_flow', 0) > 0]
        f1s = [m['f1'] for m in results[s].values()
               if 'f1' in m and m.get('n_flow', 0) > 0]
        results[s]['_summary'] = {
            'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
            'mean_f1': float(np.mean(f1s)) if f1s else None,
            'n_splits': len(aucs),
            'setting': s,
            'chrono_series': chrono_series,
            'chrono_min_train': min_train,
        }
        with open(paths[s], 'w') as f:
            json.dump(results[s], f, indent=2)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL EXTRAPOLATION (POOLED) SUMMARY")
    print("=" * 60)
    for s in SETTINGS:
        print(f"  [{s}] {results[s]['_summary']}")
        print(f"    Saved: {paths[s]}")

    return results


def chronological_extrapolation_pooled_trainthresh_contrastive(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files=None,
    max_workers: int = 1,
) -> Dict:
    """
    Strict counterpart to `chronological_extrapolation_pooled_contrastive`:
    identical splits, identical encoder training, identical GMM fit — the
    ONLY difference is the classification threshold. Here it is estimated
    exclusively from training-NoFlow GMM scores (mean + 3*sigma, computed
    once per split) and then applied FIXED to every pooled future test
    patch, with zero use of any future series' own NoFlow reference. This
    is the "no calibration period available at deployment" bound, as
    opposed to the per-series-calibrated protocol in the sibling function
    (which assumes each new site's own initial NoFlow window is known).

    ROC-AUC is threshold-independent, so per-split AUC here is expected to
    equal the calibrated-protocol AUC for the same split — only F1/accuracy/
    confusion matrix are expected to move (usually worse, since a single
    training-time threshold can't track per-series baseline shifts).

    Results are written to NEW files (`chrono_pooled_{setting}_trainthresh_
    results.json`) in the same results dir as the calibrated protocol —
    the existing `chrono_pooled_{setting}_results.json` files are never
    read, written, or otherwise touched by this function.
    """
    cfg = config['contrastive'].get('encoder', {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root = cfg.get('output_root', './contrastive')
    data_root = cfg.get('data_root', './prepared')
    results_dir = data_path / output_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = results_dir / 'optuna_best_params.json'
    if best_params_path.exists():
        with open(best_params_path, 'r') as f:
            best = json.load(f)
        tuned = best.get('params', {})
        print(f"Using tuned hyperparameters from {best_params_path}")
        _apply_variant_from_best_params(cfg, best)
    else:
        tuned = {}
        print("No Optuna best params found — using config defaults.")

    chrono_series = cfg.get('chrono_series') or cfg.get('loso_series', [
        '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
        '7.03.2026', '15.03.2026',
    ])
    min_train = cfg.get('chrono_min_train', 4)
    chrono_test_series = cfg.get('chrono_test_series', None)
    temp_target = cfg.get('gmm_temp_target', 'temp_pipe')
    temp_target = temp_target if isinstance(temp_target, str) else temp_target[0]

    if min_train >= len(chrono_series) - 1:
        raise ValueError(
            f"chrono_min_train ({min_train}) leaves no room for a non-empty "
            f"test range with {len(chrono_series)} series."
        )

    splits = [
        (chrono_series[:k], chrono_series[k:])
        for k in range(min_train, len(chrono_series))
    ]
    if chrono_test_series is not None:
        splits = [(tr, te) for tr, te in splits if te[0] in chrono_test_series]

    splits_data, metadata_by_id, prepared_base = _load_splits_and_metadata(data_path, data_root)

    encoder_type = cfg.get('encoder_type', 'autoencoder')
    encoder_arch = cfg.get('encoder_arch', 'cnn')

    def _hp_for_run() -> Dict:
        _is_transformer = (encoder_arch == 'transformer')
        return {
            'encoder_type': encoder_type,
            'encoder_arch': encoder_arch,
            'cnn_channels': None if _is_transformer else cfg.get('cnn_channels', [32, 64, 128]),
            'cnn_kernel_size': None if _is_transformer else cfg.get('cnn_kernel_size', 3),
            'dropout': cfg.get('dropout', 0.3),
            'latent_dim': tuned.get('latent_dim', cfg.get('latent_dim', 32)),
            'projection_dim': tuned.get('projection_dim', cfg.get('projection_dim', 64)),
            'k_per_series': tuned.get('k_per_series', cfg.get('k_per_series', 16)),
            'contrastive_loss': cfg.get('contrastive_loss', 'supcon'),
            'lambda_supcon': tuned.get('lambda_supcon', cfg.get('lambda_supcon', 0.5)),
            'supcon_temperature': tuned.get('supcon_temperature', cfg.get('supcon_temperature', 0.1)),
            'lambda_rnc': tuned.get('lambda_rnc', cfg.get('lambda_rnc', 0.5)),
            'rnc_temperature': tuned.get('rnc_temperature', cfg.get('rnc_temperature', 2.0)),
            'rnc_similarity': cfg.get('rnc_similarity', 'cosine'),
            'lambda_deepsad': tuned.get('lambda_deepsad', cfg.get('lambda_deepsad', 1.0)),
            'deepsad_margin': tuned.get('deepsad_margin', cfg.get('deepsad_margin', 1.0)),
            'deepsad_repulsion_weight': cfg.get('deepsad_repulsion_weight', 1.0),
            'lr': tuned.get('lr', cfg.get('lr', 1e-3)),
            'd_model': tuned.get('d_model', cfg.get('d_model', 128)),
            'num_heads': tuned.get('num_heads', cfg.get('num_heads', 4)),
            'num_layers': tuned.get('num_layers', cfg.get('num_layers', 4)),
            'ff_dim': tuned.get('ff_dim', cfg.get('ff_dim', 256)),
            'transformer_dropout': tuned.get('transformer_dropout', cfg.get('transformer_dropout', 0.1)),
            'warmup_epochs': tuned.get('warmup_epochs', cfg.get('warmup_epochs', 0)),
        }

    regress_map = _regress_target_map_if_needed(data_path, cfg, metadata_by_id)
    selected = _resolve_selected_features(cfg)

    SETTINGS = ['none', 'pipe']
    results: Dict[str, Dict[str, Dict]] = {s: {} for s in SETTINGS}
    paths: Dict[str, Path] = {
        s: results_dir / f'chrono_pooled_{s}_trainthresh_results.json' for s in SETTINGS
    }
    for s in SETTINGS:
        if paths[s].exists():
            with open(paths[s]) as f:
                results[s] = json.load(f)
            results[s].pop('_summary', None)
    if results['none']:
        print(f"Resuming: loaded {len(results['none'])} existing splits from {paths['none']}")

    for train_series, test_series_list in splits:
        split_key = f"train1-{len(train_series)}_test{len(train_series) + 1}-{len(chrono_series)}"
        if split_key in results['none']:
            print(f"\n[skip] {split_key} — already in results")
            continue

        print("\n" + "=" * 60)
        print(f"CHRONOLOGICAL EXTRAPOLATION (POOLED, TRAIN-THRESHOLD-ONLY): train on "
              f"{len(train_series)} series ({train_series[0]} .. {train_series[-1]}) -> "
              f"test {len(test_series_list)} series ({test_series_list[0]} .. {test_series_list[-1]})")
        print("=" * 60)

        meta_common = {
            'n_train_series': len(train_series),
            'train_series': train_series,
            'n_test_series': len(test_series_list),
            'test_series_range': test_series_list,
        }

        all_ids = _ids_for_series(splits_data, metadata_by_id, train_series + test_series_list)
        train_side_ids = [sid for sid in all_ids if metadata_by_id[sid]['series'] in train_series]
        test_ids = [sid for sid in all_ids if metadata_by_id[sid]['series'] in test_series_list]

        data = _prepare_dataset_tensors(
            train_ids=train_side_ids,
            val_ids=[],
            test_ids=test_ids,
            prepared_base=prepared_base,
            metadata_by_id=metadata_by_id,
            selected_features=selected,
            scaler_type=cfg.get('scaler', 'standard'),
            dual=(cfg.get('cnn_channel_mode', 'dual') == 'dual'),
            regress_target_map=regress_map,
        )

        hp = _hp_for_run()
        model, _history, _best_ep = _run_single_training(
            data, hp, device,
            epochs=cfg.get('epochs', 80),
            patience=cfg.get('patience', 15),
            verbose=False,
        )

        gmm_n = cfg.get('gmm_n_components', 1)
        gmm_cov = cfg.get('gmm_covariance_type', 'full')

        if len(data['test_patches']) == 0 or len(data['train_patches']) == 0:
            print("  (no data)")
            for s in SETTINGS:
                results[s][split_key] = dict(meta_common)
        else:
            train_labels = np.asarray(data['train_labels'])
            train_noflow_mask = train_labels == 0
            train_latents_all = _extract_latents(model, data['train_patches'], device)
            train_noflow_lat = train_latents_all[train_noflow_mask]

            test_latents = _extract_latents(model, data['test_patches'], device)
            test_labels = np.asarray(data['test_labels'])
            test_series_arr = np.array(data['test_series'])

            if train_noflow_mask.sum() < max(gmm_n * 2, 4):
                print("  (not enough train NoFlow patches)")
                for s in SETTINGS:
                    results[s][split_key] = dict(meta_common)
            else:
                # --- Setting "none": plain global GMM, fixed training-only threshold ---
                try:
                    gmm_none = GaussianMixture(
                        n_components=gmm_n, covariance_type=gmm_cov,
                        reg_covar=1e-3, random_state=42,
                    )
                    gmm_none.fit(train_noflow_lat)
                    scores_none = -gmm_none.score_samples(test_latents)
                    train_scores_none = -gmm_none.score_samples(train_noflow_lat)
                    threshold_none = float(train_scores_none.mean() + 3.0 * train_scores_none.std())
                    m_none = _pooled_metrics_fixed_threshold(test_labels, scores_none, threshold_none)
                    m_none.update(meta_common)
                    m_none['gmm_fit_n_noflow'] = int(train_noflow_mask.sum())
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  [none] fit failed: {type(e).__name__}: {e}")
                    m_none = dict(meta_common)
                results['none'][split_key] = m_none
                roc = m_none.get('roc_auc')
                print(f"  [none] pooled ROC-AUC: {roc:.4f}" if roc is not None else "  [none] (skipped)")

                # --- Setting "pipe": temp_pipe-conditioned GMM, fixed training-only threshold ---
                unique_series = sorted(set(list(data['train_series']) + list(data['test_series'])))
                temp_map_pipe = _compute_per_series_regress_target(data_path, unique_series, temp_target)
                train_series_arr = np.array(data['train_series'])
                train_temps = np.array(
                    [temp_map_pipe.get(s, np.nan) for s in train_series_arr], dtype=np.float64,
                )
                valid_train = ~np.isnan(train_temps)
                fit_mask = train_noflow_mask & valid_train
                n_unique_T = len(np.unique(train_temps[fit_mask])) if fit_mask.sum() else 0

                if fit_mask.sum() < max(gmm_n * 2, 4) or n_unique_T < 2:
                    print("  [pipe] insufficient training temperature data, skipping")
                    results['pipe'][split_key] = dict(meta_common)
                else:
                    T_fit = train_temps[fit_mask].reshape(-1, 1)
                    z_fit = train_latents_all[fit_mask]
                    A = np.hstack([T_fit, np.ones_like(T_fit)])
                    coef, *_ = np.linalg.lstsq(A, z_fit, rcond=None)
                    slope, intercept = coef[0], coef[1]
                    residual_train = z_fit - (T_fit @ slope[None, :] + intercept[None, :])
                    try:
                        gmm_pipe = GaussianMixture(
                            n_components=gmm_n, covariance_type=gmm_cov,
                            reg_covar=1e-3, random_state=42,
                        )
                        gmm_pipe.fit(residual_train)
                        train_scores_pipe = -gmm_pipe.score_samples(residual_train)
                        threshold_pipe = float(train_scores_pipe.mean() + 3.0 * train_scores_pipe.std())

                        scores_pipe = np.full(len(test_labels), np.nan)
                        for s in sorted(set(test_series_arr)):
                            mask = test_series_arr == s
                            T_s = temp_map_pipe.get(s, np.nan)
                            if np.isnan(T_s):
                                continue
                            z_pred_s = T_s * slope + intercept
                            residual_s = test_latents[mask] - z_pred_s[None, :]
                            scores_pipe[mask] = -gmm_pipe.score_samples(residual_s)
                        valid_test = ~np.isnan(scores_pipe)
                        if valid_test.sum() == 0 or len(set(test_labels[valid_test].tolist())) < 2:
                            print("  [pipe] no valid pooled test scores, skipping")
                            results['pipe'][split_key] = dict(meta_common)
                        else:
                            m_pipe = _pooled_metrics_fixed_threshold(
                                test_labels[valid_test], scores_pipe[valid_test], threshold_pipe,
                            )
                            m_pipe.update(meta_common)
                            m_pipe['gmm_fit_n_noflow'] = int(fit_mask.sum())
                            m_pipe['temp_target'] = temp_target
                            results['pipe'][split_key] = m_pipe
                            roc = m_pipe.get('roc_auc')
                            print(f"  [pipe] pooled ROC-AUC: {roc:.4f}" if roc is not None else "  [pipe] (skipped)")
                    except (np.linalg.LinAlgError, ValueError) as e:
                        print(f"  [pipe] fit failed: {type(e).__name__}: {e}")
                        results['pipe'][split_key] = dict(meta_common)

        # Save after every split, both settings.
        for s in SETTINGS:
            with open(paths[s], 'w') as f:
                json.dump(results[s], f, indent=2)

    for s in SETTINGS:
        aucs = [m['roc_auc'] for m in results[s].values()
                if 'roc_auc' in m and m.get('n_flow', 0) > 0]
        f1s = [m['f1'] for m in results[s].values()
               if 'f1' in m and m.get('n_flow', 0) > 0]
        results[s]['_summary'] = {
            'mean_roc_auc': float(np.mean(aucs)) if aucs else None,
            'mean_f1': float(np.mean(f1s)) if f1s else None,
            'n_splits': len(aucs),
            'setting': s,
            'chrono_series': chrono_series,
            'chrono_min_train': min_train,
            'protocol': 'training_threshold_only',
        }
        with open(paths[s], 'w') as f:
            json.dump(results[s], f, indent=2)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL EXTRAPOLATION (POOLED, TRAIN-THRESHOLD-ONLY) SUMMARY")
    print("=" * 60)
    for s in SETTINGS:
        print(f"  [{s}] {results[s]['_summary']}")
        print(f"    Saved: {paths[s]}")

    return results
