"""
Relaxation Module - Step 3 of Preprocessing Pipeline.

This module extracts thermal relaxation segments from cleaned data:
1. Aggregate points to block means (64 points -> 1 block)
2. Detect jumps on block means -> relaxation boundaries
3. Extract continuous segments between jumps
4. Add relaxation_id and valid_relaxation columns in-place

Reads from and overwrites ``data/processed/<series>/<file>.csv``.

Based on Notebook 3: [3] Extract Relaxations.ipynb
"""
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ..common.plotting import plot_relaxations_overview


def aggregate_to_blocks(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    block_size: int = 64
) -> pd.DataFrame:
    """
    Aggregate point-level data to block-level by computing block means.

    This is the key preprocessing step: instead of working with ~100,000 points,
    we work with ~1,500 block means. This naturally smooths out noise and makes
    jump detection much more robust.

    Args:
        df: DataFrame with valid points only.
        value_col: Column containing measurement values.
        time_col: Column containing timestamps.
        block_size: Number of points per block (default: 64).

    Returns:
        DataFrame with one row per block containing:
        - block_id: Block index
        - block_mean: Mean value of the block
        - block_std: Standard deviation within block
        - block_time: Timestamp of block center
        - start_idx: Start index in original df
        - end_idx: End index in original df
        - n_points: Number of points in block
    """
    n_points = len(df)
    n_blocks = n_points // block_size

    if n_blocks == 0:
        return pd.DataFrame()

    blocks = []
    values = df[value_col].values
    times = df[time_col].values

    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size

        block_values = values[start_idx:end_idx]
        block_times = times[start_idx:end_idx]

        blocks.append({
            'block_id': i,
            'block_mean': np.mean(block_values),
            'block_std': np.std(block_values),
            'block_min': np.min(block_values),
            'block_max': np.max(block_values),
            'block_time': block_times[block_size // 2],  # Center timestamp
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_points': block_size
        })

    return pd.DataFrame(blocks)


def detect_jumps_mad(
    block_means: np.ndarray,
    threshold_factor: float = 5.0
) -> Tuple[np.ndarray, float, float]:
    """
    Detect jumps in block means using MAD (Median Absolute Deviation).

    Algorithm:
    1. Compute block-to-block differences
    2. Calculate robust threshold using MAD
    3. Flag blocks where difference exceeds threshold

    Args:
        block_means: Array of block mean values.
        threshold_factor: Multiplier for MAD-based threshold.

    Returns:
        Tuple of:
        - jump_block_indices: Block indices where jumps START
        - threshold: The computed threshold value
        - robust_std: The robust standard deviation estimate
    """
    if len(block_means) < 2:
        return np.array([], dtype=int), 0.0, 0.0

    # Block-to-block differences
    diffs = np.diff(block_means)

    # Robust statistics using MAD
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))
    robust_std = 1.4826 * mad  # Scale factor for normal distribution

    # Threshold
    threshold = threshold_factor * robust_std

    if threshold == 0:
        # All diffs are identical (unlikely), no jumps
        return np.array([], dtype=int), 0.0, 0.0

    # Detect jumps: where |diff - median| > threshold
    jump_positions = np.where(np.abs(diffs - median_diff) > threshold)[0]

    # jump_positions[i] means: jump between block i and block i+1
    # So the NEW segment starts at block i+1
    jump_block_indices = jump_positions + 1

    return jump_block_indices, threshold, robust_std


def detect_jumps_iqr(
    block_means: np.ndarray,
    iqr_factor: float = 3.0
) -> Tuple[np.ndarray, float, float]:
    """
    Detect jumps in block means using IQR (Interquartile Range).

    Args:
        block_means: Array of block mean values.
        iqr_factor: Multiplier for IQR-based threshold.

    Returns:
        Tuple of (jump_block_indices, threshold, iqr)
    """
    if len(block_means) < 2:
        return np.array([], dtype=int), 0.0, 0.0

    diffs = np.diff(block_means)

    q1, q3 = np.percentile(diffs, [25, 75])
    iqr = q3 - q1
    threshold = iqr_factor * iqr

    if threshold == 0:
        return np.array([], dtype=int), 0.0, 0.0

    median_diff = np.median(diffs)
    jump_positions = np.where(np.abs(diffs - median_diff) > threshold)[0]
    jump_block_indices = jump_positions + 1

    return jump_block_indices, threshold, iqr


def compute_relaxation_slope(
    times: np.ndarray,
    values: np.ndarray,
    method: str = 'linear_regression'
) -> float:
    """
    Compute the slope of a thermal relaxation curve.

    A slope near zero indicates thermal equilibrium.
    A negative slope indicates cooling (relaxation in progress).

    Args:
        times: Array of timestamps.
        values: Array of measurement values.
        method: 'linear_regression' or 'endpoints'.

    Returns:
        Slope in units/second.
    """
    if len(values) < 2:
        return 0.0

    # Convert times to numeric (seconds from first point)
    if isinstance(times[0], (pd.Timestamp, datetime, np.datetime64)):
        t_numeric = pd.to_datetime(times).astype(np.int64) / 1e9
    else:
        t_numeric = np.array(times, dtype=float)

    t_numeric = t_numeric - t_numeric[0]

    if method == 'linear_regression':
        slope, _, _, _, _ = stats.linregress(t_numeric, values)
        return slope
    elif method == 'endpoints':
        dt = t_numeric[-1] - t_numeric[0]
        if dt == 0:
            return 0.0
        return (values[-1] - values[0]) / dt
    else:
        raise ValueError(f"Unknown slope method: {method}")


def add_relaxation_columns(
    df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[dict]]:
    """
    Add relaxation_id and valid_relaxation columns to the DataFrame.

    Algorithm:
    1. Filter to valid points (valid == 1)
    2. Aggregate to block means (64 points -> 1 block)
    3. Detect jumps on block means -> relaxation boundaries
    4. Add columns back to original DataFrame

    Args:
        df: DataFrame from clean step (with 'valid' column).
        config: relaxation config from relaxation.json.

    Returns:
        Tuple of:
        - df: Original DataFrame with added columns
        - df_blocks: Block-level DataFrame (for plotting)
        - jump_block_indices: Array of block indices where jumps occur
        - metadata_list: List of relaxation metadata dicts
    """
    value_col = config['value_col']
    time_col = config['time_col']
    relaxation_cfg = config['relaxation_detection']

    min_length = relaxation_cfg['min_relaxation_length']  # In POINTS
    jump_config = relaxation_cfg['jump_detection']
    slope_config = relaxation_cfg['slope_filter']
    block_size = jump_config.get('block_size', 64)

    # Minimum blocks = min_length / block_size
    min_blocks = max(1, min_length // block_size)

    # Initialize new columns
    df = df.copy()
    df['relaxation_id'] = -1  # -1 = not in any relaxation
    df['valid_relaxation'] = 0

    # 1. Sort and filter to valid points
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    valid_mask = df_sorted['valid'] == 1
    valid_indices = df_sorted.index[valid_mask].values  # Original indices of valid points
    df_valid = df_sorted[valid_mask].reset_index(drop=True)

    empty_result = (df_sorted, pd.DataFrame(), np.array([]), [])

    if len(df_valid) < min_length:
        return empty_result

    # 2. Aggregate to blocks
    df_blocks = aggregate_to_blocks(df_valid, value_col, time_col, block_size)

    if len(df_blocks) < 2:
        return empty_result

    block_means = df_blocks['block_mean'].values

    # 3. Detect jumps on block means
    if not jump_config.get('enabled', True):
        jump_block_indices = np.array([], dtype=int)
    else:
        method = jump_config.get('method', 'block_mad')

        if method == 'block_mad':
            threshold_factor = jump_config.get('threshold_factor', 5.0)
            jump_block_indices, _, _ = detect_jumps_mad(block_means, threshold_factor)
        elif method == 'iqr':
            iqr_factor = jump_config.get('iqr_factor', 3.0)
            jump_block_indices, _, _ = detect_jumps_iqr(block_means, iqr_factor)
        else:
            raise ValueError(f"Unknown jump detection method: {method}")

    # 4. Segment into relaxations (at block level)
    n_blocks = len(df_blocks)
    split_points = np.concatenate([[0], jump_block_indices, [n_blocks]])
    split_points = np.unique(split_points)

    # Assign relaxation_id to blocks
    df_blocks['relaxation_id'] = -1

    metadata_list = []
    slope_method = slope_config.get('method', 'linear_regression')
    max_abs_slope = slope_config.get('max_abs_slope_per_point') if slope_config.get('enabled', False) else None

    relaxation_id = 0

    for i in range(len(split_points) - 1):
        start_block = split_points[i]
        end_block = split_points[i + 1]
        n_segment_blocks = end_block - start_block

        # Check minimum length (in blocks)
        if n_segment_blocks < min_blocks:
            continue

        # Get block-level data for this segment
        segment_blocks = df_blocks.iloc[start_block:end_block]

        # Get point-level indices for this segment (in df_valid)
        start_point_idx = int(segment_blocks['start_idx'].iloc[0])
        end_point_idx = int(segment_blocks['end_idx'].iloc[-1])

        # Compute slope on block means (more robust than point-level)
        block_times = segment_blocks['block_time'].values
        block_vals = segment_blocks['block_mean'].values
        slope = compute_relaxation_slope(block_times, block_vals, slope_method)

        # Optional slope filter
        if max_abs_slope is not None and abs(slope) > max_abs_slope:
            continue

        # Assign relaxation_id to blocks
        for block_idx in range(start_block, end_block):
            df_blocks.loc[block_idx, 'relaxation_id'] = relaxation_id

        # Map back to original DataFrame indices
        original_indices = valid_indices[start_point_idx:end_point_idx]
        df_sorted.loc[original_indices, 'relaxation_id'] = relaxation_id
        df_sorted.loc[original_indices, 'valid_relaxation'] = 1

        # Get segment points for metadata
        segment_points = df_valid.iloc[start_point_idx:end_point_idx]

        # Compute metadata
        metadata = {
            'relaxation_id': relaxation_id,
            'start_block': int(start_block),
            'end_block': int(end_block),
            'n_blocks': int(n_segment_blocks),
            'n_points': len(segment_points),
            'start_time': segment_points[time_col].iloc[0],
            'end_time': segment_points[time_col].iloc[-1],
            'duration_seconds': (segment_points[time_col].iloc[-1] - segment_points[time_col].iloc[0]).total_seconds(),
            'mean_value': float(block_vals.mean()),
            'std_value': float(block_vals.std()),
            'slope': float(slope),
            'min_value': float(block_vals.min()),
            'max_value': float(block_vals.max()),
        }

        metadata_list.append(metadata)
        relaxation_id += 1

    return df_sorted, df_blocks, jump_block_indices, metadata_list


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def process_single_file(
    file_path: Path,
    series_name: str,
    relaxation_config: dict,
    img_path: Optional[Path],
    plot_enabled: bool,
) -> str:
    """
    Process a single file for thermal relaxation extraction.

    Adds relaxation_id and valid_relaxation columns, overwrites CSV.

    Returns:
        A status message string.
    """
    file_name = file_path.stem
    value_col = relaxation_config['value_col']

    # Load
    df = pd.read_csv(file_path, index_col=False)
    df['Time'] = pd.to_datetime(df['Time'])

    # Add relaxation columns
    df_result, df_blocks, jump_indices, metadata_list = add_relaxation_columns(df, relaxation_config)

    # Plotting
    if plot_enabled and img_path is not None and len(df_blocks) > 0:
        series_img_dir = img_path / 'relaxations' / series_name
        leakage_times = (None, None)
        if 'flow_type' in df_result.columns:
            flow_mask = df_result['flow_type'] == 'flow'
            if flow_mask.any():
                flow_times = df_result.loc[flow_mask, 'Time']
                leakage_times = (flow_times.min(), flow_times.max())

        overview_path = series_img_dir / f"{file_name}.png"
        plot_relaxations_overview(df_blocks, jump_indices, metadata_list, f"{series_name} - {file_name}", value_col, overview_path, leakage_times=leakage_times)

    # Overwrite the same file
    save_data(df_result, file_path)

    n_relax = len(metadata_list)
    gc.collect()
    return f"[PROCESSED] {file_name} ({n_relax} relaxations)"


def process_series(
    series_cfg: dict,
    relaxation_config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1
) -> None:
    """
    Process all CSV files of a series for thermal relaxation extraction.

    Reads from processed/<series>/*.csv, adds columns, overwrites.

    Args:
        series_cfg: Config for this series.
        relaxation_config: Relaxation config from relaxation.json.
        data_path: Base data path.
        img_path: Path for saving plots.
        plot: Whether to create plots (overridden by config if present).
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers.
    """
    series_name = series_cfg['name']
    processed_root = relaxation_config['processed_root']
    plot_enabled = relaxation_config.get('plotting', {}).get('enabled', plot)

    input_dir = data_path / processed_root / series_name

    if not input_dir.exists():
        tqdm.write(f"  [SKIP] {input_dir} does not exist")
        return

    csv_files = sorted(input_dir.glob("*.csv"))
    if only_files:
        csv_files = [f for f in csv_files if f.stem in only_files]
    if not csv_files:
        return

    desc = f"  {series_name}"
    if max_workers > 1 and len(csv_files) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, fp, series_name, relaxation_config, img_path, plot_enabled): fp
                for fp in csv_files
            }
            with tqdm(total=len(csv_files), desc=desc, unit="file") as pbar:
                for future in as_completed(futures):
                    status_msg = future.result()
                    tqdm.write(f"    {status_msg}")
                    pbar.update(1)
    else:
        for file_path in tqdm(csv_files, desc=desc, unit="file"):
            status_msg = process_single_file(file_path, series_name, relaxation_config, img_path, plot_enabled)
            tqdm.write(f"    {status_msg}")


def process_relaxation(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1
) -> None:
    """
    Main entry point for relaxation processing.

    Args:
        config: Full config dict (must contain 'relaxation').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot: Whether to create plots (can be overridden by config).
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers.
    """
    relaxation_config = config['relaxation']
    plot_enabled = relaxation_config.get('plotting', {}).get('enabled', plot)

    print("=" * 60)
    print("EXTRACT THERMAL RELAXATIONS")
    print("=" * 60)
    print(f"Root:    {relaxation_config['processed_root']}")
    print(f"Series:  {[s['name'] for s in relaxation_config['series']]}")
    print(f"Plotting: {'ENABLED' if plot_enabled else 'DISABLED'}")
    print(f"Workers: {max_workers}")
    print("=" * 60)

    for series in relaxation_config['series']:
        if series.get('processed', False):
            print(f"\n[SKIP] {series['name']} (already processed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"SERIES: {series['name']}")
        print(f"{'=' * 60}")
        process_series(series, relaxation_config, data_path, img_path, plot_enabled, only_files=only_files, max_workers=max_workers)

    # --- Statistics ---
    from src.common.stats import compute_relaxation_stats, print_stats_table, save_stats
    series_names = [s['name'] for s in relaxation_config['series'] if not s.get('processed', False)]
    time_col = relaxation_config.get('time_col', 'Time')
    relaxation_stats = compute_relaxation_stats(data_path, relaxation_config['processed_root'], series_names, time_col)
    print_stats_table(relaxation_stats, "relaxation")
    save_stats(relaxation_stats, data_path / "stats" / "relaxation_stats.json")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
