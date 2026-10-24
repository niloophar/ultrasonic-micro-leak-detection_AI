"""
Data Preparation Module - Step 4 of Preparation Pipeline.

Extracts patch-building logic from the training module into a standalone
preparation step. Supports activatable "boxes" (Patch, Timeseries) with
their own parameter groups.

Pipeline position: extract -> features -> splitting -> **prepare** -> train

Overlap is only applied to training segments (data augmentation).
Validation and test segments use overlap=0.0 so each time step is
evaluated exactly once.

Output:
- Per-segment .npy files under data/classification/prepared/<series>/
- Updated metadata with preparation info (n_patches per segment)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


SCALING_MODES = (
    "none",              # 0: no prep-time scaling (train.py applies its scaler)
    "noflow_global",     # A: StandardScaler fit on all train NoFlow segments
    "noflow_per_series", # B: per-series StandardScaler on all NoFlow of that series
    "segment_center",    # C: subtract per-segment mean (drift removal)
    "time_diff",         # D: first-order time difference (drift removal)
)


# ---------------------------------------------------------------------------
# Core data preparation functions (extracted from train.py)
# ---------------------------------------------------------------------------

def downsample_segment(
    feature_array: np.ndarray,
    factor: int,
    stats: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Downsample by aggregating every `factor` consecutive points.

    Args:
        feature_array: Shape (n_points, n_features).
        factor: Number of points to aggregate.
        stats: List of statistics to compute per block.
            Supported: "mean", "std". Default: ["mean"].
            Results are concatenated along the feature axis:
            [mean_f0, ..., mean_fN, std_f0, ..., std_fN].

    Returns:
        Shape (n_points // factor, n_features * len(stats)).
    """
    if stats is None:
        stats = ["mean"]

    n_points, n_features = feature_array.shape
    n_blocks = n_points // factor
    if n_blocks == 0:
        return np.empty((0, n_features * len(stats)))

    truncated = feature_array[:n_blocks * factor]
    reshaped = truncated.reshape(n_blocks, factor, n_features)

    parts = []
    for stat in stats:
        if stat == "mean":
            parts.append(reshaped.mean(axis=1))
        elif stat == "std":
            parts.append(reshaped.std(axis=1))
        else:
            raise ValueError(f"Unknown downsample stat: {stat!r}. Use 'mean' or 'std'.")

    return np.concatenate(parts, axis=1)


def cut_patches(
    downsampled: np.ndarray,
    patch_width: int,
    overlap: float = 0.0,
) -> np.ndarray:
    """
    Cut downsampled segment into fixed-width patches.

    Args:
        downsampled: Shape (n_steps, n_features).
        patch_width: Number of time steps per patch.
        overlap: Fraction of overlap between patches (0.0 = no overlap).

    Returns:
        Shape (n_patches, n_features, patch_width).
    """
    n_steps, n_features = downsampled.shape
    if n_steps < patch_width:
        return np.empty((0, n_features, patch_width))

    stride = max(1, int(patch_width * (1.0 - overlap)))
    starts = list(range(0, n_steps - patch_width + 1, stride))

    patches = np.zeros((len(starts), n_features, patch_width))
    for i, s in enumerate(starts):
        patches[i] = downsampled[s:s + patch_width].T

    return patches


def center_patches(
    patches: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Center each patch independently along the time axis.

    Args:
        patches: Shape (N, F, W) — N patches, F features, W time steps.
        method: One of 'none', 'mean', 'zscore'.
            - 'none':   no centering, return as-is.
            - 'mean':   subtract per-patch, per-feature mean.
            - 'zscore': subtract mean and divide by std per-patch, per-feature.

    Returns:
        Centred patches with the same shape (N, F, W).
    """
    if method == 'none' or len(patches) == 0:
        return patches

    # mean per patch per feature: (N, F, 1)
    mu = patches.mean(axis=2, keepdims=True)

    if method == 'mean':
        return patches - mu

    if method == 'zscore':
        sigma = patches.std(axis=2, keepdims=True)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (patches - mu) / sigma

    raise ValueError(f"Unknown centering method: {method!r}. Use 'none', 'mean', or 'zscore'.")


# ---------------------------------------------------------------------------
# Prep-time scaling (fit on NoFlow, apply globally or per series)
# ---------------------------------------------------------------------------

def _time_mask(timestamps: pd.Series, start_hour: int, end_hour: int) -> np.ndarray:
    """Return a boolean mask for rows whose hour-of-day is in [start_hour, end_hour).

    Supports wrap-around (e.g. [22, 6] = late-night 22:00–05:59).
    """
    hours = pd.to_datetime(timestamps).dt.hour
    if start_hour == end_hour:
        return np.ones(len(hours), dtype=bool)
    if start_hour < end_hour:
        return ((hours >= start_hour) & (hours < end_hour)).to_numpy()
    return ((hours >= start_hour) | (hours < end_hour)).to_numpy()


def _load_and_filter_segment_rows(
    meta: dict,
    features_base: Path,
    cache: Dict[str, pd.DataFrame],
    time_filter_cfg: Optional[dict] = None,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Load a segment, strip the Time passthrough, apply the NoFlow time filter.

    The Time column is added by feature_engineering.py as an auxiliary column
    and must be removed before downsampling — it is not a feature. If a
    NoFlow time filter is active and the segment has ``label==0``, rows whose
    hour-of-day falls outside the window are dropped here. Flow segments
    (``label==1``) and segments without a Time column pass through unchanged.

    Returns ``None`` if the segment is missing on disk, empty, or filtered to
    zero rows. Otherwise returns ``(values, feature_names)`` where
    ``feature_names`` lists the surviving (non-Time, non-segment_id) columns.
    """
    series = meta['series']
    seg_id = meta['segment_id']
    label = meta.get('label', -1)

    if series not in cache:
        series_csv = features_base / f"{series}.csv"
        if not series_csv.exists():
            return None
        cache[series] = pd.read_csv(series_csv, index_col=False)
    df = cache[series]
    seg_df = df[df['segment_id'] == seg_id].drop(
        columns=['segment_id']
    ).reset_index(drop=True)

    time_series = None
    if 'Time' in seg_df.columns:
        time_series = seg_df['Time']
        seg_df = seg_df.drop(columns=['Time'])

    if (time_filter_cfg is not None
            and time_filter_cfg.get('enabled', False)
            and label == 0
            and time_series is not None):
        start_h = int(time_filter_cfg.get('start_hour', 0))
        end_h = int(time_filter_cfg.get('end_hour', 5))
        mask = _time_mask(time_series, start_h, end_h)
        if not mask.any():
            return None
        seg_df = seg_df.loc[mask].reset_index(drop=True)

    if len(seg_df) == 0:
        return None
    return seg_df.values, list(seg_df.columns)


def _load_segment_downsampled(
    meta: dict,
    features_base: Path,
    downsample_factor: int,
    downsample_stats: List[str],
    cache: Dict[str, pd.DataFrame],
    time_filter_cfg: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Load a segment's features from CSV and return its downsampled array."""
    result = _load_and_filter_segment_rows(
        meta, features_base, cache, time_filter_cfg,
    )
    if result is None:
        return None
    values, _ = result
    if len(values) < downsample_factor:
        return None
    return downsample_segment(
        values, downsample_factor, stats=downsample_stats,
    )


def _fit_prep_scaler(
    mode: str,
    metadata: List[dict],
    features_base: Path,
    patch_cfg: dict,
    segment_labels: Dict[int, int],
    train_ids: Optional[set],
    time_filter_cfg: Optional[dict] = None,
) -> Union[StandardScaler, Dict[str, StandardScaler], None]:
    """
    Fit scaler(s) for NoFlow-based scaling modes.

    - 'noflow_global': one StandardScaler fit on all train NoFlow segments.
    - 'noflow_per_series': one StandardScaler per series, fit on ALL NoFlow
        segments of that series (train and test splits combined). This
        simulates realistic per-site calibration.

    Returns None if mode is 'none', 'segment_center' or 'time_diff'.
    """
    if mode not in ("noflow_global", "noflow_per_series"):
        return None

    downsample_factor = patch_cfg.get('downsample_factor', 16)
    downsample_stats = patch_cfg.get('downsample_stats', ['mean'])
    cache: Dict[str, pd.DataFrame] = {}

    def _is_noflow(seg_id: int) -> bool:
        return segment_labels.get(seg_id, -1) == 0

    if mode == "noflow_global":
        accumulated = []
        for meta in metadata:
            seg_id = meta['segment_id']
            if train_ids is not None and seg_id not in train_ids:
                continue
            if not _is_noflow(seg_id):
                continue
            arr = _load_segment_downsampled(
                meta, features_base, downsample_factor, downsample_stats, cache,
                time_filter_cfg=time_filter_cfg,
            )
            if arr is not None and len(arr) > 0:
                accumulated.append(arr)
        if not accumulated:
            print("  [WARN] No train NoFlow data found — scaler fit skipped.")
            return None
        full = np.concatenate(accumulated, axis=0)
        scaler = StandardScaler()
        scaler.fit(full)
        print(f"  Global NoFlow scaler fitted on {len(full)} samples "
              f"x {full.shape[1]} features")
        return scaler

    # noflow_per_series
    series_arrays: Dict[str, List[np.ndarray]] = {}
    for meta in metadata:
        seg_id = meta['segment_id']
        if not _is_noflow(seg_id):
            continue
        arr = _load_segment_downsampled(
            meta, features_base, downsample_factor, downsample_stats, cache,
        )
        if arr is not None and len(arr) > 0:
            series_arrays.setdefault(meta['series'], []).append(arr)

    scalers: Dict[str, StandardScaler] = {}
    for series, arrays in sorted(series_arrays.items()):
        full = np.concatenate(arrays, axis=0)
        scaler = StandardScaler()
        scaler.fit(full)
        scalers[series] = scaler
        print(f"  Per-series NoFlow scaler [{series}]: "
              f"{len(full)} samples x {full.shape[1]} features")
    if not scalers:
        print("  [WARN] No NoFlow data found per series — scaler fit skipped.")
        return None
    return scalers


def _apply_prep_scaling(
    downsampled: np.ndarray,
    mode: str,
    scaler_data: Union[StandardScaler, Dict[str, StandardScaler], None],
    series: str,
) -> np.ndarray:
    """Apply the prep-time scaling / transformation to a downsampled segment."""
    if mode == "none" or len(downsampled) == 0:
        return downsampled

    if mode == "noflow_global":
        if scaler_data is None:
            return downsampled
        return scaler_data.transform(downsampled)

    if mode == "noflow_per_series":
        if not isinstance(scaler_data, dict):
            return downsampled
        scaler = scaler_data.get(series)
        if scaler is None:
            return downsampled
        return scaler.transform(downsampled)

    if mode == "segment_center":
        # Per-segment: subtract the mean over time (per feature).
        return downsampled - downsampled.mean(axis=0, keepdims=True)

    if mode == "time_diff":
        # First-order difference along time; prepend first row to keep shape.
        first = downsampled[:1]
        return np.diff(downsampled, axis=0, prepend=first)

    return downsampled


# ---------------------------------------------------------------------------
# Patch preparation
# ---------------------------------------------------------------------------

def _prepare_patches(
    patch_cfg: dict,
    metadata: List[dict],
    features_base: Path,
    out_base: Path,
    train_ids: Optional[set] = None,
) -> List[dict]:
    """
    Prepare patches for all segments: downsample -> cut -> center -> save .npy.

    Overlap is only applied to training segments (augmentation).
    Validation and test segments always use overlap=0.0.

    Args:
        patch_cfg: Patch box config (downsample_factor, patch_width, etc.).
        metadata: List of segment metadata dicts from features/metadata.json.
        features_base: Path to features/ directory.
        out_base: Path to prepared/ output directory.
        train_ids: Set of segment IDs belonging to the train split.
            Only these get the configured overlap; all others get 0.0.
            If None, all segments get the configured overlap (legacy).

    Returns:
        List of updated metadata dicts with preparation info added.
    """
    downsample_factor = patch_cfg.get('downsample_factor', 64)
    patch_width = patch_cfg.get('patch_width', 32)
    patch_overlap = patch_cfg.get('patch_overlap', 0.0)
    centering = patch_cfg.get('centering', 'none')
    downsample_stats = patch_cfg.get('downsample_stats', ['mean'])
    scaling_mode = patch_cfg.get('scaling_mode', 'none')
    time_filter_cfg = patch_cfg.get('noflow_time_filter', {}) or {}
    if scaling_mode not in SCALING_MODES:
        print(f"  [WARN] Unknown scaling_mode '{scaling_mode}' — using 'none'.")
        scaling_mode = 'none'

    if time_filter_cfg.get('enabled', False):
        s_h = int(time_filter_cfg.get('start_hour', 0))
        e_h = int(time_filter_cfg.get('end_hour', 5))
        print(f"  NoFlow time filter: keep rows with hour in "
              f"[{s_h:02d}:00, {e_h:02d}:00) — applied to label=0 segments only.")

    # Build segment_id -> label lookup for scaler fitting
    segment_labels: Dict[int, int] = {
        m['segment_id']: m['label'] for m in metadata
    }

    # Fit prep-time scaler(s) if requested
    scaler_data = None
    if scaling_mode in ("noflow_global", "noflow_per_series"):
        print(f"  Fitting prep-time scaler (mode: {scaling_mode})...")
        scaler_data = _fit_prep_scaler(
            scaling_mode, metadata, features_base, patch_cfg,
            segment_labels, train_ids, time_filter_cfg=time_filter_cfg,
        )

    updated_metadata = []
    total_patches = 0
    train_patches_count = 0
    eval_patches_count = 0
    filtered_out_count = 0

    # Cache loaded series CSVs to avoid re-reading per segment
    _series_cache: Dict[str, pd.DataFrame] = {}

    for meta in tqdm(metadata, desc="  Preparing patches", unit="seg"):
        seg_id = meta['segment_id']
        series = meta['series']

        # Overlap only for training segments
        if train_ids is not None:
            is_train = seg_id in train_ids
            overlap = patch_overlap if is_train else 0.0
        else:
            is_train = True
            overlap = patch_overlap

        loaded = _load_and_filter_segment_rows(
            meta, features_base, _series_cache,
            time_filter_cfg=time_filter_cfg,
        )
        if loaded is None:
            # Either CSV missing, segment empty, or entirely outside the
            # NoFlow time window — silent unless debugging would help later.
            filtered_out_count += 1
            continue
        feature_array, feature_names = loaded

        if len(feature_array) < downsample_factor:
            print(f"    [SKIP] Segment {seg_id}: too few points "
                  f"({len(feature_array)} < {downsample_factor})")
            continue

        # Downsample
        downsampled = downsample_segment(
            feature_array, downsample_factor, stats=downsample_stats,
        )

        # Expand feature names for additional stats
        base_names = list(feature_names)
        expanded_names = list(base_names)
        for stat in downsample_stats[1:]:
            expanded_names += [f"{n}__{stat}" for n in base_names]
        feature_names = expanded_names
        if len(downsampled) < patch_width:
            print(f"    [SKIP] Segment {seg_id}: too few blocks after downsample "
                  f"({len(downsampled)} < {patch_width})")
            continue

        # Apply prep-time scaling / transformation (A/B/C/D)
        if scaling_mode != "none":
            downsampled = _apply_prep_scaling(
                downsampled, scaling_mode, scaler_data, series,
            )

        # Cut patches (overlap only for train)
        patches = cut_patches(downsampled, patch_width, overlap)
        if len(patches) == 0:
            continue

        # Center
        if centering != 'none':
            patches = center_patches(patches, centering)

        # Save .npy
        series_dir = out_base / series
        series_dir.mkdir(parents=True, exist_ok=True)
        patch_file = f"segment_{seg_id}_patches.npy"
        np.save(series_dir / patch_file, patches)

        # Update metadata
        updated_meta = dict(meta)
        updated_meta['patch_file'] = patch_file
        updated_meta['n_patches'] = int(patches.shape[0])
        updated_meta['n_features'] = int(patches.shape[1])
        updated_meta['patch_width'] = int(patches.shape[2])
        updated_meta['feature_names'] = feature_names
        updated_meta['prep_scaling_mode'] = scaling_mode
        updated_metadata.append(updated_meta)

        total_patches += len(patches)
        if is_train:
            train_patches_count += len(patches)
        else:
            eval_patches_count += len(patches)

    print(f"\n  Total segments prepared: {len(updated_metadata)}")
    print(f"  Total patches created:  {total_patches}")
    if filtered_out_count > 0:
        print(f"  Segments skipped (empty / outside time window): "
              f"{filtered_out_count}")
    if train_ids is not None:
        print(f"    Train patches (overlap={patch_overlap}): {train_patches_count}")
        print(f"    Val/Test patches (overlap=0.0):  {eval_patches_count}")

    return updated_metadata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_prepare(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> Dict:
    """
    Run data preparation step: load feature CSVs and prepare data for training.

    Supports activatable boxes:
    - **Patch**: downsample -> cut patches -> center -> save per-segment .npy
    - **Timeseries**: placeholder for future time-series preparation

    Args:
        config: Full config dict (must contain 'classification.data_preparation').
        data_path: Base data path.
        img_path: Path for saving plots (unused).
        plot_enabled: Whether to create plots (unused).
        only_files: Optional (unused).
        max_workers: Number of workers (unused).

    Returns:
        Summary dict with preparation results.
    """
    prep_cfg = config['preparation']['data_preparation']
    output_root = prep_cfg.get('output_root', './prepared')

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Load feature metadata
    class_base = data_path / output_root
    features_base = class_base / 'features'
    meta_path = features_base / 'metadata.json'

    if not meta_path.exists():
        print(f"[ERROR] Feature metadata not found: {meta_path}")
        print("Run 'features' step first.")
        return {}

    with open(meta_path, 'r') as f:
        all_metadata = json.load(f)

    print(f"  Segments found: {len(all_metadata)}")

    # Load splits.json to determine train segments (overlap only for train)
    splits_path = class_base / 'splits.json'
    train_ids = None
    if splits_path.exists():
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)
        train_ids = set(splits_data.get('splits', {}).get('train', []))
        print(f"  Splits loaded: {len(train_ids)} train segments "
              f"(overlap only for train)")
    else:
        print("  [WARN] splits.json not found — applying overlap to all segments.")
        print("         Run 'splitting' before 'prepare' for correct behaviour.")

    results = {}

    # --- Patch box ---
    patch_cfg = prep_cfg.get('patch', {})
    if patch_cfg.get('enabled', False):
        patch_overlap = patch_cfg.get('patch_overlap', 0.0)
        print(f"\n--- PATCH PREPARATION ---")
        print(f"  Downsample factor: {patch_cfg.get('downsample_factor', 64)}")
        print(f"  Patch width:       {patch_cfg.get('patch_width', 32)}")
        print(f"  Patch overlap:     {patch_overlap} (train only)")
        print(f"  Centering:         {patch_cfg.get('centering', 'none')}")
        print(f"  Downsample stats:  {patch_cfg.get('downsample_stats', ['mean'])}")
        print(f"  Scaling mode:      {patch_cfg.get('scaling_mode', 'none')}")
        _tf = patch_cfg.get('noflow_time_filter', {}) or {}
        if _tf.get('enabled', False):
            print(f"  NoFlow time filter: "
                  f"[{int(_tf.get('start_hour', 0)):02d}:00, "
                  f"{int(_tf.get('end_hour', 5)):02d}:00)")
        else:
            print(f"  NoFlow time filter: disabled")

        prepared_dir_name = patch_cfg.get('prepared_dir', 'prepared')
        prepared_base = class_base / prepared_dir_name
        prepared_base.mkdir(parents=True, exist_ok=True)

        updated_metadata = _prepare_patches(
            patch_cfg, all_metadata, features_base, prepared_base,
            train_ids=train_ids,
        )

        # Merge with kept metadata from previous runs and save
        prep_meta_path = prepared_base / 'metadata.json'
        existing_prep_metadata = []
        if prep_meta_path.exists():
            with open(prep_meta_path, 'r') as f:
                existing_prep_metadata = json.load(f)

        # Determine which series were processed in this run
        current_series = set(m['series'] for m in updated_metadata)
        kept_prep_metadata = [m for m in existing_prep_metadata if m['series'] not in current_series]
        merged_prep_metadata = kept_prep_metadata + updated_metadata

        with open(prep_meta_path, 'w') as f:
            json.dump(merged_prep_metadata, f, indent=2)

        print(f"  Metadata saved: {prep_meta_path}")

        results['patch'] = {
            'segments_prepared': len(updated_metadata),
            'total_patches': sum(m['n_patches'] for m in updated_metadata),
        }

    # --- Timeseries box (placeholder) ---
    ts_cfg = prep_cfg.get('timeseries', {})
    if ts_cfg.get('enabled', False):
        print(f"\n--- TIMESERIES PREPARATION ---")
        print("  [INFO] Timeseries preparation not yet implemented.")
        results['timeseries'] = {'status': 'not_implemented'}

    print(f"\n{'=' * 60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}")

    return results
