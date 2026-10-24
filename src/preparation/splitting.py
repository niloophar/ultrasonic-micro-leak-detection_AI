"""
Splitting Module - Step 3 of Classification Pipeline.

Assigns segments to train/val/test splits at the segment level.
No data transformation happens here — that is done in the training module.

Output: splits.json with segment_id -> train/val/test mapping.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _stratified_split(
    segment_ids: List[int],
    segment_labels: Dict[int, int],
    segment_series: Dict[int, str],
    fractions: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Stratified split of segment IDs into arbitrary named splits.

    Stratification is applied per (series × class) to ensure every split
    contains samples from every series AND every class (when possible).
    This avoids degenerate cases like "all leakage from series X landed
    in train, none in val" that occur with per-class-only stratification.

    Fractions are relative to each (series, class) bucket. The last split
    gets the remainder to avoid rounding issues. A bucket with only 1
    segment is placed in the first split (train).

    Args:
        segment_ids: List of segment IDs to split.
        segment_labels: Mapping segment_id -> label (0 or 1).
        segment_series: Mapping segment_id -> series name.
        fractions: Ordered dict of split_name -> fraction (must sum to ~1.0).
            Example: {"train": 0.7, "val": 0.1, "test": 0.2}
        seed: Random seed.

    Returns:
        Dict of split_name -> list of segment IDs.
    """
    rng = np.random.RandomState(seed)
    split_names = list(fractions.keys())

    # Group by (series, class)
    buckets: Dict[Tuple[str, int], List[int]] = {}
    for seg_id in segment_ids:
        key = (segment_series[seg_id], segment_labels[seg_id])
        buckets.setdefault(key, []).append(seg_id)

    result: Dict[str, List[int]] = {name: [] for name in split_names}

    # Deterministic ordering over buckets
    for key in sorted(buckets.keys()):
        bucket_segs = buckets[key]
        if len(bucket_segs) <= 1:
            # With only 1 segment, keep it in the first split (train)
            result[split_names[0]].extend(bucket_segs)
            continue

        rng.shuffle(bucket_segs)
        offset = 0
        for i, name in enumerate(split_names):
            if i == len(split_names) - 1:
                # Last split gets the remainder
                result[name].extend(bucket_segs[offset:])
            else:
                n = max(1, int(len(bucket_segs) * fractions[name]))
                result[name].extend(bucket_segs[offset:offset + n])
                offset += n

    return result


def process_splitting(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> Dict:
    """
    Assign segments to train/val/test splits and save splits.json.

    Reads feature metadata to know which segments exist and their labels.

    Args:
        config: Full config dict (must contain 'classification').
        data_path: Base data path.
        img_path: Path for saving plots (unused).
        plot_enabled: Whether to create plots (unused).
        only_files: Optional (unused).
        max_workers: Number of workers (unused).

    Returns:
        Summary dict with split counts.
    """
    split_config = config['preparation']['splitting']
    output_root = split_config['output_root']
    val_fraction = split_config.get('val_fraction', 0.2)
    seed = split_config.get('seed', 42)

    train_series = split_config.get('train_series', [])
    test_series = split_config.get('test_series', [])
    test_fraction = split_config.get('test_fraction', 0.2)

    print("=" * 60)
    print("SPLITTING")
    print("=" * 60)
    print(f"Train series:      {train_series}")
    print(f"Test series:       {test_series}")
    if not test_series:
        print(f"  -> No test series: stratified {1-val_fraction-test_fraction:.0%}/"
              f"{val_fraction:.0%}/{test_fraction:.0%} split from train series")
    print(f"Val fraction:      {val_fraction}")
    print(f"Test fraction:     {test_fraction} {'(unused — test from separate series)' if test_series else ''}")
    print(f"Seed:              {seed}")
    print("=" * 60)

    # Load feature metadata
    features_base = data_path / output_root / 'features'
    meta_path = features_base / 'metadata.json'
    if not meta_path.exists():
        print(f"[ERROR] Feature metadata not found: {meta_path}")
        print("Run 'features' step first.")
        return {}

    with open(meta_path, 'r') as f:
        all_metadata = json.load(f)

    # Separate blind segments (label=-1, from "blind": true series in label.json)
    # from labeled segments. Blind segments go to blind_test only; they must
    # never leak into train/val/test where labeled code would see them.
    blind_ids: List[int] = [m['segment_id'] for m in all_metadata if m.get('label') == -1]
    all_metadata = [m for m in all_metadata if m.get('label') != -1]
    if blind_ids:
        print(f"  Blind segments (label=-1): {len(blind_ids)} → blind_test bucket")

    # Build segment_id -> label and segment_id -> series mappings
    segment_labels: Dict[int, int] = {}
    segment_series: Dict[int, str] = {}
    for meta in all_metadata:
        segment_labels[meta['segment_id']] = meta['label']
        segment_series[meta['segment_id']] = meta['series']

    # Collect train segment IDs
    train_all_ids = [
        m['segment_id'] for m in all_metadata
        if m['series'] in train_series
    ]

    if not train_all_ids:
        print("[ERROR] No training segments found.")
        return {}

    if test_series:
        # Explicit test series: split train into train/val only
        test_ids = [
            m['segment_id'] for m in all_metadata
            if m['series'] in test_series
        ]
        splits = _stratified_split(
            train_all_ids, segment_labels, segment_series,
            fractions={"train": 1.0 - val_fraction, "val": val_fraction},
            seed=seed,
        )
        train_ids = splits["train"]
        val_ids = splits["val"]
    else:
        # No test series: stratified train/val/test from train_series
        train_frac = 1.0 - val_fraction - test_fraction
        splits = _stratified_split(
            train_all_ids, segment_labels, segment_series,
            fractions={"train": train_frac, "val": val_fraction, "test": test_fraction},
            seed=seed,
        )
        train_ids = splits["train"]
        val_ids = splits["val"]
        test_ids = splits["test"]

    # Build lookup dicts from metadata
    metadata_by_id = {m['segment_id']: m for m in all_metadata}

    def _label_dist(ids):
        n_leak = sum(1 for i in ids if segment_labels.get(i) == 1)
        n_no = sum(1 for i in ids if segment_labels.get(i) == 0)
        return n_leak, n_no

    def _total_points(ids):
        return sum(metadata_by_id[i]['n_points'] for i in ids)

    def _print_split(name, ids, indent=2):
        n_leak, n_no = _label_dist(ids)
        n_pts = _total_points(ids)
        prefix = " " * indent
        print(f"{prefix}{name}: {len(ids)} segments "
              f"(leakage={n_leak}, no_leakage={n_no}), "
              f"{n_pts:,} points")

        # Per-series breakdown
        series_ids: Dict[str, List[int]] = {}
        for seg_id in ids:
            s = metadata_by_id[seg_id]['series']
            series_ids.setdefault(s, []).append(seg_id)
        for series_name in sorted(series_ids):
            s_ids = series_ids[series_name]
            s_leak, s_no = _label_dist(s_ids)
            s_pts = _total_points(s_ids)
            print(f"{prefix}  {series_name}: {len(s_ids)} segments "
                  f"(leakage={s_leak}, no_leakage={s_no}), "
                  f"{s_pts:,} points")

    print()
    _print_split("Train", train_ids)
    _print_split("Val", val_ids)
    _print_split("Test", test_ids)

    # Save splits.json
    splits_output = {
        'train_series': train_series,
        'test_series': test_series,
        'val_fraction': val_fraction,
        'seed': seed,
        'splits': {
            'train': sorted(train_ids),
            'val': sorted(val_ids),
            'test': sorted(test_ids),
            'blind_test': sorted(blind_ids),
        },
    }

    splits_dir = data_path / output_root
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / 'splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits_output, f, indent=2)

    print(f"\n  Splits saved: {splits_path}")

    # --- Statistics ---
    from src.common.stats import build_splitting_stats, print_stats_table, save_stats
    # Read patch config from data_preparation config
    prep_cfg = config['preparation'].get('data_preparation', {})
    patch_cfg = prep_cfg.get('patch', {})
    split_stats = build_splitting_stats(
        train_ids, val_ids, test_ids, segment_labels, metadata_by_id,
        downsample_factor=patch_cfg.get('downsample_factor', 64),
        patch_width=patch_cfg.get('patch_width', 32),
        patch_overlap=patch_cfg.get('patch_overlap', 0.0),
    )
    print_stats_table(split_stats, "splitting")
    save_stats(split_stats, data_path / "stats" / "splitting_stats.json")

    print(f"\n{'=' * 60}")
    print("SPLITTING COMPLETE")
    print(f"{'=' * 60}")

    return {
        'train': len(train_ids),
        'val': len(val_ids),
        'test': len(test_ids),
    }
