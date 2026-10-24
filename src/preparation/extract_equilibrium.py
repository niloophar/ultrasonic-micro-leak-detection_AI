"""
Extract Equilibrium Data Module - Step 1 of Classification Pipeline.

Reads preprocessed CSV files and extracts equilibrium data points
for the two classes: flow (leakage, label=1) and noflow (no-leakage, label=0).
Excludes 'undefined' flow types.

Invalid points (valid!=1) are filtered out upfront before segment detection.
This eliminates index gaps caused by outlier points and ensures that
contiguous equilibrium regions on block-level remain contiguous on
point-level.
"""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common.constants import UP_COLS, DOWN_COLS, TEMP_COLS


# Columns to keep in extracted segments
FEATURE_COLS = UP_COLS + DOWN_COLS + [
    'amplitude_up in mV',
    'amplitude_down in mV',
    'tdiff in ps',
]

LABEL_MAP = {'noflow': 0, 'preleakage': 0, 'postleakage': 0, 'flow': 1, 'unknown': -1}


def extract_from_file(
    file_path: Path,
    time_col: str = 'Time',
) -> List[dict]:
    """
    Extract equilibrium segments from a single preprocessed CSV.

    Invalid points are removed first, then contiguous runs of
    ``equilibrium==1`` with the same binary label form segments.

    Args:
        file_path: Path to preprocessed CSV file.
        time_col: Name of the time column.

    Returns:
        List of segment dicts with keys:
            - data: DataFrame with feature columns + Time + label
            - label: 0 (noflow) or 1 (flow)
            - flow_type: original flow_type string
            - file_name: source file stem
            - n_points: number of valid data points
            - start_time: segment start timestamp
            - end_time: segment end timestamp
    """
    file_name = file_path.stem

    # Empty / malformed CSVs (zero-byte files, header-only files, partial
    # writes from a crashed preprocessing run) should be skipped quietly
    # instead of aborting the whole extract pass.
    try:
        df = pd.read_csv(file_path, index_col=False)
    except pd.errors.EmptyDataError:
        print(f"  [skip] {file_path.name}: empty CSV")
        return []
    if df.empty:
        print(f"  [skip] {file_path.name}: no data rows")
        return []

    if time_col not in df.columns:
        print(f"  [skip] {file_path.name}: time column {time_col!r} missing")
        return []
    df[time_col] = pd.to_datetime(df[time_col])

    # Check required columns
    required = ['equilibrium', 'valid', 'flow_type']
    for col in required:
        if col not in df.columns:
            return []

    # Filter to valid points first — removes outliers and eliminates
    # index gaps that would otherwise fragment equilibrium regions.
    df = df[df['valid'] == 1].reset_index(drop=True)

    # Map flow_type to binary label (NaN for excluded types like 'undefined')
    df['_label'] = df['flow_type'].map(LABEL_MAP)

    # Segment boundary mask: equilibrium==1 and known flow_type
    eq_mask = (df['equilibrium'] == 1) & df['_label'].notna()

    if not eq_mask.any():
        return []

    # Find contiguous segments: gap > 1 in index OR label change.
    # Since invalid points are already removed and index is reset,
    # a gap > 1 means there are genuine non-equilibrium points in between.
    eq_indices = df.index[eq_mask].values
    eq_labels = df.loc[eq_mask, '_label'].values.astype(int)

    segments = []  # list of (start_pos, end_pos) into eq_indices
    seg_start = 0
    for i in range(1, len(eq_indices)):
        index_gap = eq_indices[i] - eq_indices[i - 1] > 1
        label_change = eq_labels[i] != eq_labels[seg_start]
        if index_gap or label_change:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, len(eq_indices)))

    results = []
    keep_cols = [time_col] + FEATURE_COLS + TEMP_COLS

    for seg_start_pos, seg_end_pos in segments:
        seg_idx = eq_indices[seg_start_pos:seg_end_pos]
        seg_df = df.loc[seg_idx].copy()

        if len(seg_df) < 2:
            continue

        flow_type = seg_df['flow_type'].iloc[0]
        label = LABEL_MAP.get(flow_type)
        if label is None:
            continue

        # Keep only feature columns + time
        available_cols = [c for c in keep_cols if c in seg_df.columns]
        seg_data = seg_df[available_cols].reset_index(drop=True)
        seg_data['label'] = label

        results.append({
            'data': seg_data,
            'label': label,
            'flow_type': flow_type,
            'file_name': file_name,
            'n_points': len(seg_data),
            'start_time': str(seg_df[time_col].iloc[0]),
            'end_time': str(seg_df[time_col].iloc[-1]),
        })

    return results


def process_extract(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> dict:
    """
    Extract equilibrium segments from all series.

    Args:
        config: Full config dict (must contain 'classification').
        data_path: Base data path.
        img_path: Path for saving plots (unused for now).
        plot_enabled: Whether to create plots.
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers (unused, sequential).

    Returns:
        Dict with extraction summary.
    """
    ext_config = config['preparation']['extract_equilibrium']
    processed_root = ext_config['processed_root']
    output_root = ext_config['output_root']
    time_col = ext_config.get('time_col', 'Time')

    all_series = ext_config.get('series', [])

    print("=" * 60)
    print("EXTRACT EQUILIBRIUM DATA")
    print("=" * 60)
    print(f"Source:    {processed_root}")
    print(f"Output:    {output_root}/segments/")
    print(f"Series:    {all_series}")
    print(f"Classes:   flow (label=1), noflow (label=0)")
    print("=" * 60)

    total_segments = 0
    total_points = {'flow': 0, 'noflow': 0, 'preleakage': 0, 'postleakage': 0, 'unknown': 0}

    # Load existing metadata to support incremental runs
    seg_dir = data_path / output_root / 'segments'
    meta_path = seg_dir / 'metadata.json'
    existing_metadata = []
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            existing_metadata = json.load(f)

    # Keep metadata for series NOT being processed now
    processed_series = set(all_series)
    kept_metadata = [m for m in existing_metadata if m['series'] not in processed_series]

    # Continue seg_id from the highest existing ID + 1
    all_existing_ids = [m['segment_id'] for m in kept_metadata]
    seg_id = (max(all_existing_ids) + 1) if all_existing_ids else 0

    all_metadata = []

    for series_name in all_series:
        input_dir = data_path / processed_root / series_name

        if not input_dir.exists():
            print(f"\n[SKIP] {input_dir} does not exist")
            continue

        csv_files = sorted(input_dir.glob("*.csv"))
        if only_files:
            csv_files = [f for f in csv_files if f.stem in only_files]
        if not csv_files:
            print(f"\n[SKIP] {series_name}: no CSV files found")
            continue

        seg_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSERIES: {series_name}")

        series_frames = []

        for file_path in tqdm(csv_files, desc=f"  {series_name}", unit="file"):
            segments = extract_from_file(file_path, time_col)

            for seg in segments:
                seg_data = seg['data'].copy()
                seg_data['segment_id'] = seg_id
                series_frames.append(seg_data)

                meta = {
                    'segment_id': seg_id,
                    'series': series_name,
                    'file_name': seg['file_name'],
                    'flow_type': seg['flow_type'],
                    'label': seg['label'],
                    'n_points': seg['n_points'],
                    'start_time': seg['start_time'],
                    'end_time': seg['end_time'],
                }
                all_metadata.append(meta)
                total_points[seg['flow_type']] += seg['n_points']
                seg_id += 1

            if segments:
                tqdm.write(
                    f"    [PROCESSED] {file_path.stem}: "
                    f"{len(segments)} segment(s)"
                )

        # Write one CSV per series
        if series_frames:
            series_csv = seg_dir / f"{series_name}.csv"
            series_df = pd.concat(series_frames, ignore_index=True)
            series_df.to_csv(series_csv, index=False)
            print(f"  Saved: {series_csv} ({len(series_frames)} segments, {len(series_df)} rows)")

    total_segments = seg_id

    # Merge with kept metadata from previous runs and save
    merged_metadata = kept_metadata + all_metadata
    with open(meta_path, 'w') as f:
        json.dump(merged_metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"Total segments: {total_segments}")
    noflow_total = total_points['noflow'] + total_points['preleakage'] + total_points['postleakage']
    print(f"Flow (leakage) points:      {total_points['flow']:,}")
    print(f"Noflow (no-leakage) points: {noflow_total:,}")
    print(f"  - noflow: {total_points['noflow']:,}, preleakage: {total_points['preleakage']:,}, postleakage: {total_points['postleakage']:,}")
    if total_points['unknown']:
        print(f"Blind (unknown) points:     {total_points['unknown']:,}")
    print(f"Metadata saved: {meta_path}")
    print(f"{'=' * 60}")

    return {
        'total_segments': total_segments,
        'total_points': total_points,
        'metadata': all_metadata,
    }
