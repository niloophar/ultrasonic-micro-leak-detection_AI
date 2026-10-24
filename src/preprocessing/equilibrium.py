"""
Thermal Equilibrium Detection Module - Step 5 of Preprocessing Pipeline.

This module detects thermally stable (equilibrium) regions in ultrasound data:
1. Sliding window linear regression to compute local slopes
2. Flag windows where |slope| < threshold as equilibrium
3. Merge consecutive equilibrium windows into segments

Processes both leakage and non-leakage data within the same file:
- Leakage: filter flow_type=='flow', valid==1, within leakage time windows
- Non-leakage: per relaxation_id, skip relaxations overlapping leakage windows

Reads from and overwrites ``data/processed/<series>/<file>.csv``,
adding an ``equilibrium`` column.

Based on Notebook 5: [5] Thermal Equilibrium.ipynb
"""
import gc
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ..common.plotting import plot_equilibrium_detection, plot_equilibrium_temperature
from .relaxation import aggregate_to_blocks


# =============================================================================
# CORE DETECTION
# =============================================================================


def detect_equilibrium_regions(
    block_means: np.ndarray,
    block_times: np.ndarray,
    window_length: int = 250,
    hop_size: int = 50,
    slope_threshold: float = 0.75,
    min_segment_length: int = 50,
    adaptive: bool = False,
    adaptive_factor: float = 2.0,
    max_gap_blocks: int = 0,
    max_rmse: Optional[float] = None,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """
    Detect equilibrium regions using sliding window linear regression.

    Algorithm:
    1. Slide a window of `window_length` blocks with `hop_size` step
    2. For each window, compute slope via scipy.stats.linregress (time in seconds)
    3. Flag windows as equilibrium:
       - Fixed mode: |slope| < slope_threshold
       - Adaptive mode: |slope| < adaptive_factor * std_err
         (slope is not statistically distinguishable from zero)
       - Additionally (if max_rmse is set): reject windows where the RMSE
         of the linear fit exceeds the threshold. This catches windows that
         span jumps/steps despite having low slope.
    4. Build a per-block boolean mask from overlapping equilibrium windows
    5. Bridge small gaps: fill gaps of <= max_gap_blocks non-eq blocks between eq blocks
    6. Merge consecutive equilibrium blocks into segments
    7. Filter segments shorter than min_segment_length

    Args:
        block_means: Array of block mean values.
        block_times: Array of block timestamps (datetime64 or Timestamp).
        window_length: Number of blocks per sliding window.
        hop_size: Step size between windows.
        slope_threshold: Maximum absolute slope (ps/s) for equilibrium (fixed mode).
        min_segment_length: Minimum number of blocks for a valid segment.
        adaptive: If True, use std_err-based threshold instead of fixed slope_threshold.
        adaptive_factor: Multiplier for std_err in adaptive mode (e.g. 2.0 ~ 95% CI).
        max_gap_blocks: Bridge gaps of this many (or fewer) non-eq blocks between eq regions.
            Useful when small recording gaps break otherwise continuous equilibrium. 0 = disabled.
        max_rmse: Maximum RMSE (root mean squared error) of the linear fit for a
            window to qualify as equilibrium. Same unit as block_means (e.g. ps).
            None = disabled.

    Returns:
        Tuple of:
        - segments: List of segment dicts with keys:
            segment_id, start_block, end_block, n_blocks,
            start_time, end_time, duration_seconds,
            mean_value, std_value, mean_slope
        - equilibrium_mask: Boolean array (len = n_blocks), True = equilibrium
        - slopes: Array of slopes per window position (len = n_windows)
    """
    n_blocks = len(block_means)

    if n_blocks < window_length:
        return [], np.zeros(n_blocks, dtype=bool), np.array([])

    # Convert timestamps to seconds from start
    times_pd = pd.to_datetime(block_times)
    t_seconds = (times_pd - times_pd[0]).total_seconds().values

    # Sliding window slopes
    window_starts = list(range(0, n_blocks - window_length + 1, hop_size))
    slopes = np.zeros(len(window_starts))
    window_is_eq = np.zeros(len(window_starts), dtype=bool)

    for i, start in enumerate(window_starts):
        end = start + window_length
        t_win = t_seconds[start:end]
        y_win = block_means[start:end]

        slope, intercept, _, _, std_err = stats.linregress(t_win, y_win)
        slopes[i] = slope

        if adaptive:
            # Slope is equilibrium if NOT statistically distinguishable from zero.
            # std_err adapts to the noise: noisy data -> higher threshold (lenient),
            # smooth data -> lower threshold (strict, rejects gradual trends).
            threshold = adaptive_factor * std_err if std_err > 0 else slope_threshold
            window_is_eq[i] = abs(slope) < threshold
        else:
            window_is_eq[i] = abs(slope) < slope_threshold

        # Goodness-of-fit check: reject windows where linear fit has large RMSE
        # (e.g. windows spanning a jump/step despite low overall slope)
        if window_is_eq[i] and max_rmse is not None:
            _, polyfit_res, _, _, _ = np.polyfit(t_win, y_win, 1, full=True)
            if len(polyfit_res) > 0:
                rmse = np.sqrt(polyfit_res[0] / len(y_win))
                if rmse > max_rmse:
                    window_is_eq[i] = False

    # Build per-block equilibrium mask from overlapping windows
    equilibrium_mask = np.zeros(n_blocks, dtype=bool)
    for i, start in enumerate(window_starts):
        if window_is_eq[i]:
            end = start + window_length
            equilibrium_mask[start:end] = True

    # Bridge small gaps: fill gaps of <= max_gap_blocks between eq regions
    if max_gap_blocks > 0:
        equilibrium_mask = _bridge_gaps(equilibrium_mask, max_gap_blocks)

    # Merge consecutive True blocks into segments
    segments = []
    segment_id = 0
    in_segment = False
    seg_start = 0

    for idx in range(n_blocks):
        if equilibrium_mask[idx] and not in_segment:
            seg_start = idx
            in_segment = True
        elif not equilibrium_mask[idx] and in_segment:
            seg_end = idx
            if (seg_end - seg_start) >= min_segment_length:
                seg = _build_segment_dict(
                    segment_id, seg_start, seg_end,
                    block_means, times_pd, t_seconds, slopes, window_starts, window_length
                )
                segments.append(seg)
                segment_id += 1
            in_segment = False

    # Handle segment that runs to the end
    if in_segment:
        seg_end = n_blocks
        if (seg_end - seg_start) >= min_segment_length:
            seg = _build_segment_dict(
                segment_id, seg_start, seg_end,
                block_means, times_pd, t_seconds, slopes, window_starts, window_length
            )
            segments.append(seg)

    return segments, equilibrium_mask, slopes


def _bridge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Bridge small gaps in a boolean mask.

    Fills runs of False values that are <= max_gap blocks long
    and are bounded by True on both sides.
    """
    result = mask.copy()
    n = len(result)
    i = 0
    while i < n:
        if not result[i]:
            # Start of a gap
            gap_start = i
            while i < n and not result[i]:
                i += 1
            gap_len = i - gap_start
            # Bridge if gap is small enough and bounded by True on both sides
            if gap_len <= max_gap and gap_start > 0 and i < n:
                result[gap_start:i] = True
        else:
            i += 1
    return result


def _build_segment_dict(
    segment_id: int,
    start_block: int,
    end_block: int,
    block_means: np.ndarray,
    times_pd: pd.DatetimeIndex,
    t_seconds: np.ndarray,
    slopes: np.ndarray,
    window_starts: list,
    window_length: int,
) -> dict:
    """Build a segment metadata dict for an equilibrium region."""
    seg_means = block_means[start_block:end_block]

    # Compute mean slope from windows that overlap this segment
    overlapping_slopes = []
    for i, ws in enumerate(window_starts):
        we = ws + window_length
        if ws < end_block and we > start_block:
            overlapping_slopes.append(slopes[i])
    mean_slope = float(np.mean(overlapping_slopes)) if overlapping_slopes else 0.0

    return {
        'segment_id': segment_id,
        'start_block': int(start_block),
        'end_block': int(end_block),
        'n_blocks': int(end_block - start_block),
        'start_time': str(times_pd[start_block]),
        'end_time': str(times_pd[end_block - 1]),
        'duration_seconds': float(t_seconds[end_block - 1] - t_seconds[start_block]),
        'mean_value': float(np.mean(seg_means)),
        'std_value': float(np.std(seg_means)),
        'mean_slope': mean_slope,
    }


# =============================================================================
# LEAKAGE TIME WINDOW UTILITIES
# =============================================================================


def get_leakage_time_windows(
    series_name: str,
    label_config: dict,
    df: Optional[pd.DataFrame] = None,
    time_col: str = 'Time',
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Extract absolute leakage time windows.

    Derives leakage dates from the DataFrame's flow_type column
    (rows where flow_type == 'flow'), then combines each unique
    date with time_of_day from the config.

    Args:
        series_name: Name of the series to look up.
        label_config: The label config from label.json.
        df: DataFrame with 'flow_type' and time_col columns. Required
            to derive leakage dates.
        time_col: Name of the timestamp column in df.

    Returns:
        List of (start_ts, end_ts) tuples representing leakage windows.
    """
    series_cfg = None
    for s in label_config.get('series', []):
        if s['name'] == series_name:
            series_cfg = s
            break

    if series_cfg is None:
        return []

    leakage_rule = series_cfg.get('leakage_rule', {})
    tod = leakage_rule.get('time_of_day', {})

    if not tod:
        return []

    t_start = datetime.strptime(tod['start'], '%H:%M').time()
    t_end = datetime.strptime(tod['end'], '%H:%M').time()

    # --- Determine leakage dates ---
    leakage_dates = set()

    # Primary: derive from DataFrame flow_type
    if df is not None and 'flow_type' in df.columns and time_col in df.columns:
        flow_times = pd.to_datetime(df.loc[df['flow_type'] == 'flow', time_col])
        leakage_dates = set(flow_times.dt.date.unique())

    if not leakage_dates:
        return []

    # Build windows: one per date
    windows = []
    for d in sorted(leakage_dates):
        win_start = pd.Timestamp(datetime.combine(d, t_start))
        win_end = pd.Timestamp(datetime.combine(d, t_end))
        windows.append((win_start, win_end))

    return windows


def _time_in_any_window(
    ts: pd.Timestamp,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    """Check if a single timestamp falls within any leakage window."""
    for win_start, win_end in windows:
        if win_start <= ts < win_end:
            return True
    return False


def relaxation_overlaps_leakage(
    rel_start_time: pd.Timestamp,
    rel_end_time: pd.Timestamp,
    leakage_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    """
    Check if a relaxation's time range overlaps any leakage window.

    Args:
        rel_start_time: Start time of the relaxation.
        rel_end_time: End time of the relaxation.
        leakage_windows: List of (start, end) leakage windows.

    Returns:
        True if ANY overlap exists.
    """
    for win_start, win_end in leakage_windows:
        # Overlap exists if relaxation starts before window ends
        # AND relaxation ends after window starts
        if rel_start_time < win_end and rel_end_time > win_start:
            return True
    return False


# =============================================================================
# BLOCK-TO-POINT EQUILIBRIUM MAPPING
# =============================================================================


def _map_eq_blocks_to_points(
    df: pd.DataFrame,
    eq_mask: np.ndarray,
    df_blocks: pd.DataFrame,
    valid_indices: np.ndarray,
) -> int:
    """
    Map equilibrium block mask back to point-level equilibrium column.

    Args:
        df: The full DataFrame (modified in-place).
        eq_mask: Boolean array (per block), True = equilibrium.
        df_blocks: Block-level DataFrame with start_idx/end_idx columns.
        valid_indices: Original df indices of the points used for aggregation.

    Returns:
        Number of points marked as equilibrium.
    """
    n_eq_points = 0
    for block_idx in range(len(df_blocks)):
        if eq_mask[block_idx]:
            start = int(df_blocks.iloc[block_idx]['start_idx'])
            end = int(df_blocks.iloc[block_idx]['end_idx'])
            point_indices = valid_indices[start:end]
            df.loc[point_indices, 'equilibrium'] = 1
            n_eq_points += len(point_indices)
    return n_eq_points


# =============================================================================
# COMMON EQUILIBRIUM DETECTION CALL
# =============================================================================


def _run_equilibrium_detection(
    df_blocks: pd.DataFrame,
    eq_config: dict,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """
    Run detect_equilibrium_regions with parameters from eq_config.

    Args:
        df_blocks: Block-level DataFrame with block_mean and block_time.
        eq_config: equilibrium_config section.

    Returns:
        Tuple of (segments, eq_mask, slopes).
    """
    # If residuals_filter_enabled is False, ignore max_rmse
    if eq_config.get('residuals_filter_enabled', True):
        max_rmse = eq_config.get('max_rmse', None)
    else:
        max_rmse = None

    return detect_equilibrium_regions(
        block_means=df_blocks['block_mean'].values,
        block_times=df_blocks['block_time'].values,
        window_length=eq_config.get('window_length', 250),
        hop_size=eq_config.get('hop_size', 50),
        slope_threshold=eq_config.get('slope_threshold', 0.75),
        min_segment_length=eq_config.get('min_segment_length', 50),
        adaptive=eq_config.get('adaptive', False),
        adaptive_factor=eq_config.get('adaptive_factor', 2.0),
        max_gap_blocks=eq_config.get('max_gap_blocks', 0),
        max_rmse=max_rmse,
    )


# =============================================================================
# PIPELINE ORCHESTRATION - Step 5
# =============================================================================


def process_single_file(
    file_path: Path,
    series_name: str,
    eq_config: dict,
    label_config: dict,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
) -> str:
    """
    Process a single file for equilibrium detection.

    Handles both leakage and non-leakage equilibrium within the same file:
    1. Leakage: flow_type=='flow', valid==1, within leakage time windows
    2. Non-leakage: per relaxation_id (>=0), skip if overlapping leakage

    Adds an 'equilibrium' column (0/1) and overwrites the CSV.

    Args:
        file_path: Path to the CSV file.
        series_name: Name of the series.
        eq_config: equilibrium_config section.
        label_config: label_config section (for leakage windows).
        img_path: Path for saving plots.
        plot_enabled: Whether to create plots.

    Returns:
        Status message string.
    """
    file_name = file_path.stem
    value_col = eq_config['value_col']
    time_col = eq_config['time_col']
    block_size = eq_config.get('block_size', 64)

    # Load CSV
    df = pd.read_csv(file_path, index_col=False)
    df[time_col] = pd.to_datetime(df[time_col])

    # Initialize equilibrium column
    df['equilibrium'] = 0

    # Get leakage time windows (derived from flow_type in data + time_of_day from config)
    leakage_windows = get_leakage_time_windows(
        series_name, label_config, df=df, time_col=time_col
    )

    leak_eq_points = 0
    nonleak_eq_points = 0

    # === LEAKAGE EQUILIBRIUM ===
    if leakage_windows and 'flow_type' in df.columns:
        if 'valid' in df.columns:
            leak_mask = (df['flow_type'] == 'flow') & (df['valid'] == 1)
        else:
            leak_mask = df['flow_type'] == 'flow'

        df_leak = df[leak_mask].sort_values(time_col).reset_index(drop=True)
        leak_indices = df.index[leak_mask].values

        if len(df_leak) > 0:
            # Further filter to leakage time windows
            in_window = df_leak[time_col].apply(
                lambda ts: _time_in_any_window(ts, leakage_windows)
            )
            window_mask = in_window.values
            df_leak_window = df_leak[window_mask].reset_index(drop=True)
            leak_window_indices = leak_indices[window_mask]

            if len(df_leak_window) >= block_size:
                df_blocks = aggregate_to_blocks(
                    df_leak_window, value_col, time_col, block_size
                )
                if len(df_blocks) > 0:
                    df_blocks['block_time'] = pd.to_datetime(df_blocks['block_time'])

                    segments, eq_mask, slopes = _run_equilibrium_detection(
                        df_blocks, eq_config
                    )

                    leak_eq_points = _map_eq_blocks_to_points(
                        df, eq_mask, df_blocks, leak_window_indices
                    )

                    # Plot leakage
                    if plot_enabled and img_path is not None:
                        base_dir = img_path / 'equilibrium' / series_name
                        data_start = df_blocks['block_time'].min()
                        data_end = df_blocks['block_time'].max()
                        file_windows = [
                            (ws, we) for ws, we in leakage_windows
                            if ws < data_end and we > data_start
                        ]
                        # Detection plot in value_col subdirectory
                        plot_equilibrium_detection(
                            df_blocks=df_blocks,
                            equilibrium_mask=eq_mask,
                            slopes=slopes,
                            eq_config=eq_config,
                            title=f"{file_name} (leakage)",
                            value_col=value_col,
                            save_path=base_dir / value_col / f"{file_name}_leakage_eq.png",
                            leakage_windows=file_windows,
                        )

                        # Temperature plot for leakage equilibrium
                        plot_equilibrium_temperature(
                            df=df,
                            time_col=time_col,
                            title=f"{file_name} (leakage)",
                            save_path=base_dir / 'temperature' / f"{file_name}_leakage_temp.png",
                        )

    # === NON-LEAKAGE EQUILIBRIUM ===
    if 'relaxation_id' not in df.columns:
        tqdm.write(f"    [WARN] {file_name}: 'relaxation_id' column missing "
                   f"(relaxation step not run?) - skipping non-leakage equilibrium")
    if 'relaxation_id' in df.columns:
        rel_ids = sorted([r for r in df['relaxation_id'].unique() if r >= 0])

        # Collect per-relaxation data for combined plot
        all_rel_blocks = []
        all_eq_masks = []
        all_slope_times = []
        all_slope_values = []

        for rel_id in rel_ids:
            if 'valid' in df.columns:
                rel_mask = (df['relaxation_id'] == rel_id) & (df['valid'] == 1)
            else:
                rel_mask = df['relaxation_id'] == rel_id
            df_rel = df[rel_mask].sort_values(time_col).reset_index(drop=True)
            valid_indices = df.index[rel_mask].values

            if len(df_rel) < block_size:
                continue

            # Check if relaxation overlaps any leakage window
            rel_start_time = pd.Timestamp(df_rel[time_col].min())
            rel_end_time = pd.Timestamp(df_rel[time_col].max())

            if leakage_windows and relaxation_overlaps_leakage(
                rel_start_time, rel_end_time, leakage_windows
            ):
                continue

            # Aggregate to blocks
            df_blocks = aggregate_to_blocks(df_rel, value_col, time_col, block_size)
            if len(df_blocks) == 0:
                continue
            df_blocks['block_time'] = pd.to_datetime(df_blocks['block_time'])

            # Detect equilibrium
            segments, eq_mask, slopes = _run_equilibrium_detection(
                df_blocks, eq_config
            )

            n_eq = _map_eq_blocks_to_points(df, eq_mask, df_blocks, valid_indices)
            nonleak_eq_points += n_eq

            # Collect for combined detection plot
            if plot_enabled:
                all_rel_blocks.append(df_blocks.reset_index(drop=True))
                all_eq_masks.append(eq_mask)
                if len(slopes) > 0:
                    hop = eq_config.get('hop_size', 50)
                    wl = eq_config.get('window_length', 250)
                    rel_times = pd.to_datetime(df_blocks['block_time'])
                    win_starts = list(range(0, len(df_blocks) - wl + 1, hop))
                    for i, wi in enumerate(win_starts):
                        center = wi + wl // 2
                        if center < len(rel_times) and i < len(slopes):
                            all_slope_times.append(rel_times.iloc[center])
                            all_slope_values.append(slopes[i])

                # Per-relaxation temperature plot (only if equilibrium found)
                if img_path is not None and eq_mask.any():
                    df_rel_full = df[df['relaxation_id'] == rel_id]
                    base_dir = img_path / 'equilibrium' / series_name
                    plot_equilibrium_temperature(
                        df=df_rel_full,
                        time_col=time_col,
                        title=f"{file_name} (rel {rel_id})",
                        save_path=base_dir / 'temperature' / f"{file_name}_rel{rel_id}_temp.png",
                    )

        # Combined non-leakage detection plot
        if plot_enabled and img_path is not None and all_rel_blocks:
            combined_blocks = pd.concat(all_rel_blocks, ignore_index=True)
            combined_eq_mask = np.concatenate(all_eq_masks)

            base_dir = img_path / 'equilibrium' / series_name

            data_start = combined_blocks['block_time'].min()
            data_end = combined_blocks['block_time'].max()
            file_windows = [
                (ws, we) for ws, we in leakage_windows
                if ws < data_end and we > data_start
            ] if leakage_windows else []

            # Detection plot in value_col subdirectory
            plot_equilibrium_detection(
                df_blocks=combined_blocks,
                equilibrium_mask=combined_eq_mask,
                slopes=np.array(all_slope_values),
                eq_config=eq_config,
                title=f"{file_name} (non-leakage)",
                value_col='block_mean',
                save_path=base_dir / value_col / f"{file_name}_nonleak_eq.png",
                leakage_windows=file_windows,
                slope_times=all_slope_times if all_slope_times else None,
            )

    # Overwrite CSV
    df.to_csv(file_path, index=False)

    gc.collect()
    total_eq = leak_eq_points + nonleak_eq_points
    return (
        f"[PROCESSED] {file_name} "
        f"(eq: {leak_eq_points} leak + {nonleak_eq_points} nonleak = {total_eq})"
    )


def process_series(
    series_cfg: dict,
    eq_config: dict,
    label_config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> None:
    """
    Process all files of a series for equilibrium detection.

    Reads from processed/<series>/*.csv, adds equilibrium column, overwrites.

    Args:
        series_cfg: Series config (name, etc.).
        eq_config: equilibrium_config section.
        label_config: label_config section (for leakage windows).
        data_path: Base data path.
        img_path: Path for saving plots.
        plot: Whether to create plots.
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers (currently sequential).
    """
    series_name = series_cfg['name']
    processed_root = eq_config['processed_root']
    plot_enabled = eq_config.get('plotting', {}).get('enabled', plot)

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
    for file_path in tqdm(csv_files, desc=desc, unit="file"):
        msg = process_single_file(
            file_path, series_name, eq_config, label_config,
            img_path, plot_enabled,
        )
        tqdm.write(f"    {msg}")


def process_equilibrium(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> None:
    """
    Main entry point for equilibrium detection (Step 5).

    Args:
        config: Full config dict (must contain 'equilibrium' and 'label').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot: Whether to create plots (can be overridden by config).
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers.
    """
    eq_config = config['equilibrium']
    label_config = config['label']
    plot_enabled = eq_config.get('plotting', {}).get('enabled', plot)

    adaptive = eq_config.get('adaptive', False)

    print("=" * 60)
    print("DETECT THERMAL EQUILIBRIUM")
    print("=" * 60)
    print(f"Root:              {eq_config['processed_root']}")
    print(f"Window:            {eq_config.get('window_length')} blocks, hop {eq_config.get('hop_size')}")
    if adaptive:
        print(f"Mode:              ADAPTIVE (factor={eq_config.get('adaptive_factor', 2.0)})")
    else:
        print(f"Slope threshold:   {eq_config.get('slope_threshold')} ps/s")
    resid_enabled = eq_config.get('residuals_filter_enabled', True)
    max_rmse = eq_config.get('max_rmse', None)
    if resid_enabled and max_rmse is not None:
        print(f"Max RMSE:          {max_rmse} ps")
    else:
        print(f"Max RMSE:          disabled")
    print(f"Min segment:       {eq_config.get('min_segment_length')} blocks")
    gap = eq_config.get('max_gap_blocks', 0)
    print(f"Max gap bridge:    {gap} blocks {'(disabled)' if gap == 0 else ''}")
    print(f"Plotting:          {'ENABLED' if plot_enabled else 'DISABLED'}")
    print(f"Series:            {[s['name'] for s in eq_config.get('series', [])]}")
    print("=" * 60)

    for series in eq_config.get('series', []):
        if series.get('processed', False):
            print(f"\n[SKIP] {series['name']} (already processed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"SERIES: {series['name']}")
        print(f"{'=' * 60}")

        process_series(
            series, eq_config, label_config, data_path,
            img_path, plot_enabled, only_files, max_workers,
        )

    # --- Statistics ---
    from src.common.stats import compute_equilibrium_stats, print_stats_table, save_stats
    series_names = [s['name'] for s in eq_config.get('series', []) if not s.get('processed', False)]
    time_col = eq_config.get('time_col', 'Time')
    eq_stats = compute_equilibrium_stats(data_path, eq_config['processed_root'], series_names, time_col)
    print_stats_table(eq_stats, "equilibrium")
    save_stats(eq_stats, data_path / "stats" / "equilibrium_stats.json")

    print(f"\n{'=' * 60}")
    print("DONE - Equilibrium Detection")
    print(f"{'=' * 60}")
