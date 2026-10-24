"""
Clean Data Module - Step 2 of Preprocessing Pipeline.

This module performs:
0. Temperature outlier detection and replacement (median imputation)
1. Global outliers: Points with extreme Z-scores
2. Local outliers: Points deviating from local trend

Reads from and overwrites ``data/processed/<series>/<file>.csv``,
adding a ``valid`` column.

Based on Notebook 2: [2] Clean Data.ipynb
"""
import gc
import json
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..common.plotting import (
    get_leakage_times,
    plot_all_outliers,
    plot_local_outliers,
    plot_inliers_averaged,
    plot_inliers_leakage,
    plot_temperature_combined,
    plot_drift_overview,
    plot_drift_overview_files,
    plot_drift_overview_subplots,
    plot_drift_overview_files_subplots,
)


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Compute a robust Z-score based on median and MAD.

    The classic Z-score (mean/std) is sensitive to outliers.
    This variant uses median and MAD (Median Absolute Deviation)
    and is therefore more robust.

    Args:
        series: Pandas Series with numeric values.

    Returns:
        Series with robust Z-scores.
    """
    med = series.median()
    mad = (series - med).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = series.std(ddof=0)
        scale = std if (np.isfinite(std) and std > 0) else 1.0
    return (series - med) / scale


def robust_mad_scale(series: pd.Series) -> float:
    """
    Compute a robust scale factor based on MAD.

    Args:
        series: Pandas Series with numeric values.

    Returns:
        Robust scale factor (1.4826 * MAD).
    """
    series = series.dropna()
    if len(series) == 0:
        return 1.0
    med = series.median()
    mad = (series - med).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = series.std(ddof=0)
        scale = std if (np.isfinite(std) and std > 0) else 1.0
    return float(scale)


def clean_temperature(
    df: pd.DataFrame,
    temp_cols: list,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Detect temperature outliers via robust Z-score, replace with nearest neighbor.

    Uses robust Z-score (median/MAD) for detection — adapts to the actual
    temperature range of each file. Replacement uses forward/backward fill
    to preserve abrupt but real temperature changes (e.g. pipe relaxation).

    Args:
        df: DataFrame with temperature columns.
        temp_cols: List of temperature column names to clean.
        z_threshold: Absolute robust Z-score threshold for outlier detection.

    Returns:
        DataFrame with cleaned temperature columns and new columns
        ``<col>_outlier`` (bool) indicating which points were replaced.
    """
    df_out = df.copy()
    for col in temp_cols:
        if col not in df_out.columns:
            continue
        series = df_out[col]
        z = robust_zscore(series)
        outlier_mask = z.abs() > z_threshold
        n_outliers = outlier_mask.sum()

        df_out[f"{col}_outlier"] = outlier_mask

        if n_outliers == 0:
            continue

        # Replace with nearest valid neighbor (ffill + bfill)
        cleaned = series.where(~outlier_mask)
        cleaned = cleaned.ffill().bfill()
        df_out.loc[outlier_mask, col] = cleaned.loc[outlier_mask]

    return df_out


def _filter_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    """
    Filter consecutive True-runs by minimum length.

    Args:
        mask: Boolean array with True/False values.
        min_length: Minimum length of a run to be kept.

    Returns:
        Boolean array, only runs >= min_length remain True.
    """
    keep = np.zeros_like(mask, dtype=bool)
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_length:
            keep[i:j] = True
        i = j
    return keep


def compute_local_outlier_blocks(
    df_in: pd.DataFrame,
    value_col: str,
    time_col: str,
    block_size: int,
    min_run_blocks: int,
    step_up_quantile: float,
    smooth_window_blocks: int,
    k_block: float,
    center_smooth: bool,
) -> pd.DataFrame:
    """
    Compute block-level statistics for local-outlier segmentation (Stage 2a of
    ``label_outliers``). Extracted so plotting/analysis code can reproduce the
    exact same block means, rolling-median reference, and good/keep-block
    classification used by the pipeline, without re-deriving the formula.

    Args:
        df_in: Inlier-only rows (global outliers already removed), sorted by
            time_col with a fresh RangeIndex (i.e. ``reset_index(drop=True)``).
        value_col: Column name of the measurement value.
        time_col: Column name of time.
        block_size: Number of data points per block.
        min_run_blocks: Minimum length of consecutive good block runs.
        step_up_quantile: Quantile for detecting jumps.
        smooth_window_blocks: Window size for rolling median smoothing.
        k_block: Tolerance factor for block deviation.
        center_smooth: True = centered smoothing, False = causal.

    Returns:
        DataFrame with one row per block: block_time, block_mean, n, diff,
        is_step, segment, smooth, resid_block, block_scale, good_block,
        keep_block.
    """
    df_in = df_in.copy()
    df_in["_block"] = df_in.index // block_size

    blocks = (df_in.groupby("_block", as_index=False)
              .agg(block_time=(time_col, "mean"),
                   block_mean=(value_col, "mean"),
                   n=(value_col, "size")))

    blocks["diff"] = blocks["block_mean"].diff()
    diff_positive = blocks["diff"].dropna()
    diff_positive = diff_positive[diff_positive > 0]

    if len(diff_positive) >= 10:
        step_threshold = float(diff_positive.quantile(step_up_quantile))
    else:
        step_threshold = float(blocks["diff"].dropna().abs().median() * 10)

    blocks["is_step"] = blocks["diff"] > step_threshold
    blocks["segment"] = blocks["is_step"].fillna(False).cumsum()

    blocks["smooth"] = np.nan
    blocks["resid_block"] = np.nan
    blocks["block_scale"] = np.nan
    blocks["good_block"] = False

    for seg, group in blocks.groupby("segment", sort=False):
        idx = group.index
        values = group["block_mean"]

        smooth = values.rolling(
            smooth_window_blocks,
            min_periods=max(5, smooth_window_blocks // 3),
            center=center_smooth
        ).median()
        smooth = smooth.bfill().ffill()

        residuals = values - smooth
        scale = robust_mad_scale(residuals)
        good = residuals.abs() <= (k_block * scale)

        blocks.loc[idx, "smooth"] = smooth.values
        blocks.loc[idx, "resid_block"] = residuals.values
        blocks.loc[idx, "block_scale"] = scale
        blocks.loc[idx, "good_block"] = good.values

    blocks["keep_block"] = False
    for seg, group in blocks.groupby("segment", sort=False):
        good_mask = group["good_block"].to_numpy(dtype=bool)
        keep = _filter_runs(good_mask, min_length=min_run_blocks)
        blocks.loc[group.index, "keep_block"] = keep

    return blocks


def label_outliers(
    df: pd.DataFrame,
    value_col: str = "t_up_0 in ps",
    time_col: str = "Time",
    z_global_low: float = -8.0,
    z_global_high: Optional[float] = None,
    block_size: int = 64,
    min_run_blocks: int = 50,
    step_up_quantile: float = 0.995,
    smooth_window_blocks: int = 21,
    k_block: float = 6.0,
    k_point: float = 8.0,
    center_smooth: bool = True,
) -> pd.DataFrame:
    """
    Two-stage outlier detection: Global and Local.

    Stage 1 - Global Outliers (valid = -1):
        Points with robust Z-score < z_global_low (or > z_global_high)

    Stage 2 - Local Outliers (valid = 0):
        a) Block level: blocks deviating from smoothed trend
        b) Point level: points deviating within good blocks

    Args:
        df: DataFrame with time series measurement data.
        value_col: Column name of the measurement value.
        time_col: Column name of time.
        z_global_low: Lower Z-score threshold for global outliers.
        z_global_high: Upper Z-score threshold for global outliers.
        block_size: Number of data points per block.
        min_run_blocks: Minimum length of consecutive good block runs.
        step_up_quantile: Quantile for detecting jumps.
        smooth_window_blocks: Window size for rolling median smoothing.
        k_block: Tolerance factor for block deviation.
        k_point: Tolerance factor for point deviation.
        center_smooth: True = centered smoothing, False = causal.

    Returns:
        DataFrame with new column "valid":
            1 = valid (inlier)
            0 = local outlier
           -1 = global outlier
    """
    df_out = df.sort_values(time_col).reset_index(drop=True).copy()
    df_out["valid"] = 1

    # === STAGE 1: Global Outliers ===
    z_global = robust_zscore(df_out[value_col])
    global_outlier_mask = (z_global < z_global_low)
    if z_global_high is not None:
        global_outlier_mask |= (z_global > z_global_high)
    df_out.loc[global_outlier_mask, "valid"] = -1

    # Only inliers for local analysis
    inlier_mask = ~global_outlier_mask
    df_in = df_out.loc[inlier_mask].copy()
    if df_in.empty:
        return df_out

    # === STAGE 2: Local Outliers ===
    df_in = df_in.sort_values(time_col).reset_index(drop=True).copy()

    # Compute block statistics (means, rolling-median reference, good/keep runs)
    blocks = compute_local_outlier_blocks(
        df_in,
        value_col=value_col,
        time_col=time_col,
        block_size=block_size,
        min_run_blocks=min_run_blocks,
        step_up_quantile=step_up_quantile,
        smooth_window_blocks=smooth_window_blocks,
        k_block=k_block,
        center_smooth=center_smooth,
    )
    df_in["_block"] = df_in.index // block_size

    # Map back to point level
    df_in = df_in.merge(blocks[["_block", "keep_block", "smooth"]], on="_block", how="left")
    df_in["keep_block"] = df_in["keep_block"].fillna(False)

    # Point-level check
    df_in["resid_point"] = df_in[value_col] - df_in["smooth"]
    scale_point = robust_mad_scale(df_in.loc[df_in["keep_block"], "resid_point"])
    df_in["good_point"] = df_in["resid_point"].abs() <= (k_point * scale_point)

    # Local outlier = block not in kept run OR point deviates too much
    local_outlier = (~df_in["keep_block"]) | (~df_in["good_point"])
    df_in["valid_local"] = np.where(local_outlier, 0, 1)

    # Map back to main dataframe
    df_out.loc[inlier_mask, "valid"] = df_in["valid_local"].values

    return df_out


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file (output from label step).

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with parsed Time column.
    """
    df = pd.read_csv(file_path, index_col=False)
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save DataFrame as CSV.

    Args:
        df: DataFrame to save.
        output_path: Target path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def get_label_series_config(label_config: dict, series_name: str) -> Optional[dict]:
    """
    Get the label_config series entry matching the given name.

    Args:
        label_config: Label config dict.
        series_name: Name of the series to find.

    Returns:
        Series config dict or None if not found.
    """
    for s in label_config['series']:
        if s['name'] == series_name:
            return s
    return None


def process_single_file(
    file_path: Path,
    series_name: str,
    clean_config: dict,
    leakage_rule: dict,
    img_path: Optional[Path],
    plot: bool
) -> str:
    """
    Process a single file: load, clean, optionally plot, and overwrite.

    Returns:
        A string indicating the processing result for logging.
    """
    file_name = file_path.stem

    # Extract parameters from config
    value_col = clean_config['value_col']
    time_col = clean_config['time_col']
    global_cfg = clean_config['global_outlier']
    local_cfg = clean_config['local_outlier']
    block_size = local_cfg['block_size']

    # 1. Load
    df = load_data(file_path)

    # Get leakage times from flow_type column (if available)
    leakage_times = get_leakage_times(df, leakage_rule)

    # 1b. Clean temperature columns (replace physically implausible values)
    temp_cfg = clean_config.get('temperature_cleaning', {})
    if temp_cfg.get('enabled', False):
        from ..common.constants import TEMP_COLS
        temp_cols = temp_cfg.get('columns', TEMP_COLS)
        df = clean_temperature(
            df,
            temp_cols=temp_cols,
            z_threshold=temp_cfg.get('z_threshold', 4.0),
        )

    # 2. Label outliers
    df_cleaned = label_outliers(
        df,
        value_col=value_col,
        time_col=time_col,
        z_global_low=global_cfg['z_low'],
        z_global_high=global_cfg['z_high'],
        block_size=block_size,
        min_run_blocks=local_cfg['min_run_blocks'],
        step_up_quantile=local_cfg['step_up_quantile'],
        smooth_window_blocks=local_cfg['smooth_window_blocks'],
        k_block=local_cfg['k_block'],
        k_point=local_cfg['k_point'],
        center_smooth=local_cfg['center_smooth'],
    )

    if plot and img_path is not None:
        plot_title = f"{series_name} - {file_name}"
        base_dir = img_path / 'clean' / series_name

        # --- value_col plots in subdirectories per plot type ---
        col_dir = base_dir / value_col

        plot_all_outliers(df_cleaned, plot_title, value_col, leakage_times,
                          col_dir / 'all' / f"{file_name}.png")
        plot_local_outliers(df_cleaned, plot_title, value_col, leakage_times,
                            col_dir / 'local' / f"{file_name}.png")
        plot_inliers_averaged(df_cleaned, plot_title, value_col, block_size, leakage_times,
                              col_dir / 'inliers' / f"{file_name}.png")
        if leakage_times[0] is not None and leakage_times[1] is not None:
            plot_inliers_leakage(df_cleaned, plot_title, value_col, block_size, leakage_times,
                                 col_dir / 'inliers_leakage' / f"{file_name}.png")

        # --- Combined temperature plots (Pipe+Ambient top, Diff bottom) ---
        temp_dir = base_dir / 'temperature'

        plot_temperature_combined(df_cleaned, plot_title, "all", leakage_times,
                                  block_size, temp_dir / 'all' / f"{file_name}.png")
        plot_temperature_combined(df_cleaned, plot_title, "local", leakage_times,
                                  block_size, temp_dir / 'local' / f"{file_name}.png")
        plot_temperature_combined(df_cleaned, plot_title, "inliers", leakage_times,
                                  block_size, temp_dir / 'inliers' / f"{file_name}.png")
        if leakage_times[0] is not None and leakage_times[1] is not None:
            plot_temperature_combined(df_cleaned, plot_title, "inliers_leakage",
                                      leakage_times, block_size,
                                      temp_dir / 'inliers_leakage' / f"{file_name}.png")

    # Overwrite the same file (in-place update)
    save_data(df_cleaned, file_path)

    gc.collect()
    return f"[PROCESSED] {file_name}"


def process_series(
    series_cfg: dict,
    clean_config: dict,
    label_config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1
) -> None:
    """
    Process all CSV files of a series.

    Reads from processed/<series>/*.csv, adds 'valid' column, overwrites.

    Args:
        series_cfg: Config for this series.
        clean_config: Global clean config with parameters.
        label_config: Label config (for leakage rules).
        data_path: Base data path.
        img_path: Path for saving plots.
        plot: Whether to create plots.
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers.
    """
    series_name = series_cfg['name']
    processed_root = clean_config['processed_root']

    # Get leakage rule from label_config
    label_series = get_label_series_config(label_config, series_name)
    leakage_rule = label_series.get('leakage_rule', {}) if label_series else {}

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
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_file, file_path, series_name, clean_config, leakage_rule, img_path, plot)
                for file_path in csv_files
            ]
            with tqdm(total=len(csv_files), desc=desc, unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    tqdm.write(f"  {result}")
                    pbar.update(1)
    else:
        # Sequential processing
        for file_path in tqdm(csv_files, desc=desc, unit="file"):
            result = process_single_file(file_path, series_name, clean_config, leakage_rule, img_path, plot)
            tqdm.write(f"  {result}")


def _load_drift_cache(cache_file: Path, block_size: int, value_col: str):
    """Load drift cache JSON. Returns None if missing or parameters don't match."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        if data.get('block_size') != block_size or data.get('value_col') != value_col:
            return None
        if 'files' not in data:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


_AMB_COL = "Temp Ambient in °C"


def _compute_drift_data_for_series(
    series_name: str, series_dir: Path, value_col: str, time_col: str, block_size: int
):
    """Compute block-averaged drift data for one series. Returns dict or None."""
    if not series_dir.exists():
        return None

    all_times = []
    all_values = []
    all_temp_amb = []
    files_data = []

    for csv_file in sorted(series_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, index_col=False)
        if 'valid' not in df.columns:
            continue
        df[time_col] = pd.to_datetime(df[time_col])
        inliers = df[df['valid'] == 1].sort_values(time_col).reset_index(drop=True)
        if inliers.empty:
            continue

        has_amb = _AMB_COL in inliers.columns

        # Block-average
        inliers['_block'] = inliers.index // block_size
        agg_kwargs = {time_col: (time_col, 'mean'), value_col: (value_col, 'mean')}
        if has_amb:
            agg_kwargs[_AMB_COL] = (_AMB_COL, 'mean')
        averaged = inliers.groupby('_block', as_index=False).agg(**agg_kwargs)

        block_times = averaged[time_col].tolist()
        block_values = averaged[value_col].tolist()
        all_times.extend(block_times)
        all_values.extend(block_values)
        if has_amb:
            all_temp_amb.extend(averaged[_AMB_COL].tolist())
        else:
            all_temp_amb.extend([None] * len(block_times))

        # Per-file summary
        file_mean_time = inliers[time_col].mean()
        file_mean_value = float(inliers[value_col].mean())
        file_mean_temp_amb = float(inliers[_AMB_COL].mean()) if has_amb else None
        files_data.append({
            'file_name': csv_file.stem,
            'mean_time': file_mean_time.isoformat(),
            'mean_value': file_mean_value,
            'mean_temp_amb': file_mean_temp_amb,
        })

    if not all_times:
        return None

    return {
        'name': series_name,
        'value_col': value_col,
        'block_size': block_size,
        'times': [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in all_times],
        'values': [float(v) for v in all_values],
        'temp_amb_values': [float(v) if v is not None else None for v in all_temp_amb],
        'files': files_data,
    }


def process_clean(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> None:
    """
    Main entry point for clean processing.

    Args:
        config: Full config dict (must contain 'clean' and 'label').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot: Whether to create plots.
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel workers.
    """
    clean_config = config['clean']
    label_config = config['label']

    print("=" * 60)
    print("CLEAN DATA PIPELINE")
    print("=" * 60)
    print(f"Root:   {clean_config['processed_root']}")
    print(f"Series: {[s['name'] for s in clean_config['series']]}")
    print(f"Workers: {max_workers}")
    print("=" * 60)

    for series in clean_config['series']:
        if series.get('processed', False):
            print(f"\n[SKIP] {series['name']} (already processed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"SERIES: {series['name']}")
        print(f"{'=' * 60}")
        process_series(
            series, clean_config, label_config, data_path, img_path, plot,
            only_files=only_files, max_workers=max_workers
        )

    # --- Statistics ---
    from src.common.stats import compute_clean_stats, print_stats_table, save_stats
    series_names = [s['name'] for s in clean_config['series'] if not s.get('processed', False)]
    clean_stats = compute_clean_stats(data_path, clean_config['processed_root'], series_names)
    print_stats_table(clean_stats, "clean")
    save_stats(clean_stats, data_path / "stats" / "clean_stats.json")

    # --- Drift Plot (independent of --plot flag) ---
    drift_cfg = clean_config.get('drift_plot', {})
    if drift_cfg.get('enabled', False) and img_path is not None:
        print(f"\n{'=' * 60}")
        print("DRIFT PLOT")
        print(f"{'=' * 60}")

        drift_block_size = drift_cfg.get('block_size', 512)
        fit_degree = drift_cfg.get('fit_degree', 3)
        value_col = drift_cfg.get('value_col', clean_config['value_col'])
        time_col = clean_config['time_col']
        processed_root = clean_config['processed_root']

        cache_dir = data_path / "stats" / "drift"
        cache_dir.mkdir(parents=True, exist_ok=True)

        series_data = []
        for series in clean_config['series']:
            series_name = series['name']
            series_dir = data_path / processed_root / series_name
            cache_file = cache_dir / f"{series_name}.json"

            # Try loading from cache; recompute if temp_amb_values is missing
            cached = _load_drift_cache(cache_file, drift_block_size, value_col)
            if cached is not None and 'temp_amb_values' in cached:
                series_data.append(cached)
                print(f"  {series_name}: {len(cached['values'])} blocks (cached)")
                continue

            # Compute fresh
            drift_data = _compute_drift_data_for_series(
                series_name, series_dir, value_col, time_col, drift_block_size
            )
            if drift_data is not None:
                # Save cache
                with open(cache_file, 'w') as f:
                    json.dump(drift_data, f, indent=2)
                series_data.append(drift_data)
                print(f"  {series_name}: {len(drift_data['values'])} blocks (computed)")

        if series_data:
            save_path = img_path / 'clean' / 'drift_overview.png'
            plot_drift_overview(series_data, value_col, fit_degree, save_path)
            print(f"  Saved: {save_path}")

            save_path_files = img_path / 'clean' / 'drift_overview_files.png'
            plot_drift_overview_files(series_data, value_col, fit_degree, save_path_files)
            print(f"  Saved: {save_path_files}")

            save_path_sub = img_path / 'clean' / 'drift_overview_subplots.png'
            plot_drift_overview_subplots(series_data, value_col, fit_degree, save_path_sub)
            print(f"  Saved: {save_path_sub}")

            save_path_files_sub = img_path / 'clean' / 'drift_overview_files_subplots.png'
            plot_drift_overview_files_subplots(series_data, value_col, fit_degree, save_path_files_sub)
            print(f"  Saved: {save_path_files_sub}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
