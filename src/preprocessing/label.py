"""
Label Data Module - Step 1 of Preprocessing Pipeline.

This module labels raw data as:
- 'flow': During leakage periods
- 'noflow': Outside leakage periods
- 'preleakage': Before first leakage
- 'postleakage': After last leakage
- 'undefined': Unknown class (not yet classified)

Output is written to ``data/processed/<series>/<file>.csv`` with columns
from ``kept_cols`` (defined in label config) plus ``sample`` and ``flow_type``.

Based on Notebook 1: [1] Label Data.ipynb
"""
import gc
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

from ..common.constants import NUMERICAL_COLS
from ..common.plotting import plot_series

# Default columns kept from raw data (28 columns)
DEFAULT_KEPT_COLS = ['Time'] + NUMERICAL_COLS


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load TSV data, fix a column name, and parse `Time` as ISO-8601 datetime.

    Args:
        file_path: Path to the TSV file.

    Returns:
        DataFrame with parsed Time column.
    """
    df = pd.read_csv(file_path, sep='\t', decimal=',', encoding='latin-1', index_col=False)
    df = df.rename(columns={'t_down_ 0 in ps': 't_down_0 in ps'})
    df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
    return df


def is_leakage(ts: datetime, leakage_rule: dict) -> bool:
    """
    Return True if `ts` falls into a leakage time-of-day window.

    Args:
        ts: Timestamp to check.
        leakage_rule: Dict with time_of_day (start/end).

    Returns:
        True if timestamp is in leakage period.
    """
    ts_time = ts.time()
    tod = leakage_rule.get('time_of_day', {})
    if tod:
        t_start = datetime.strptime(tod['start'], '%H:%M').time()
        t_end = datetime.strptime(tod['end'], '%H:%M').time()
        if t_start <= ts_time < t_end:
            return True
    return False


def classify_flow_types(
    time_series: pd.Series,
    leakage_rule: dict,
) -> pd.Series:
    """
    Classify each timestamp as flow/noflow/preleakage/postleakage.

    For leakage files, timestamps within the time_of_day window are 'flow',
    timestamps before the first flow are 'preleakage', after the last are
    'postleakage', and the rest are 'noflow'.

    Args:
        time_series: Series of timestamps.
        leakage_rule: Leakage rule config with time_of_day.

    Returns:
        Categorical series with values 'flow', 'preleakage', 'postleakage', 'noflow'.
    """
    leak_mask = time_series.apply(lambda t: is_leakage(t, leakage_rule))

    # Default
    flow_type = pd.Series('noflow', index=time_series.index)

    # If no leakage anywhere -> keep all as noflow
    if not leak_mask.any():
        return flow_type

    # Leakage exists -> split noflow into pre/post relative to first/last leakage timestamp
    t_first = time_series.loc[leak_mask].min()
    t_last = time_series.loc[leak_mask].max()

    flow_type.loc[leak_mask] = 'flow'
    flow_type.loc[(~leak_mask) & (time_series < t_first)] = 'preleakage'
    flow_type.loc[(~leak_mask) & (time_series > t_last)] = 'postleakage'

    return flow_type


def save_df(
    df: pd.DataFrame,
    series_name: str,
    output_root: str,
    file_name: str,
    data_path: Path
) -> None:
    """
    Save DataFrame to processed/<series>/<file>.csv.

    Args:
        df: DataFrame to save.
        series_name: Name of the series.
        output_root: Output root directory (e.g. "processed").
        file_name: Name of the file (without extension).
        data_path: Base data path.
    """
    if not df.empty:
        out_dir = data_path / output_root / series_name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{file_name}.csv", index=False)


def process_single_file(
    file_path: Path,
    series_cfg: dict,
    label_config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    file_flow_types: Optional[dict] = None
) -> str:
    """
    Process a single file: load, classify, optionally plot, and save.

    Args:
        file_path: Path to the raw TSV file.
        series_cfg: Series configuration dict.
        label_config: Label config from label.json.
        data_path: Base data path.
        img_path: Path for saving plots (optional).
        plot: Whether to create plots.
        file_flow_types: Dict mapping filename stems to flow types
            (from series config 'files' key).

    Returns:
        A string indicating the processing result for logging.
    """
    file_name = file_path.stem
    leakage_rule = series_cfg.get('leakage_rule', {})

    # Load data
    df = load_data(file_path)

    # Keep only relevant columns (from config or default)
    kept_cols = label_config.get('kept_cols', DEFAULT_KEPT_COLS)
    df = df[kept_cols]

    # Add sample column
    df['sample'] = series_cfg['name']

    # Blind mode: no labels available — assign 'unknown' to all rows and skip
    # filename/leakage-rule logic entirely. The preparation pipeline routes
    # 'unknown' segments into the blind_test bucket in splits.json.
    if series_cfg.get('blind', False):
        df['flow_type'] = 'unknown'
        if plot and img_path is not None:
            plot_series(df, title=file_name, series_name=series_cfg['name'], img_path=img_path)
        save_df(df, series_cfg['name'], label_config['output_root'],
                file_name=file_name, data_path=data_path)
        gc.collect()
        return f"[BLIND] {file_name}"

    if file_flow_types is not None and file_name in file_flow_types:
        ft = file_flow_types[file_name]
        if ft == 'undefined':
            df['flow_type'] = 'undefined'
            result_msg = f"[UNDEFINED] {file_name}"
        elif ft == 'no_leakage':
            df['flow_type'] = 'noflow'
            result_msg = f"[NO_LEAKAGE] {file_name}"
        elif ft == 'leakage':
            df['flow_type'] = classify_flow_types(df['Time'], leakage_rule)
            result_msg = f"[LEAKAGE] {file_name}"
        else:
            warnings.warn(
                f"Unknown flow type '{ft}' for file '{file_name}', "
                f"setting to 'undefined'"
            )
            df['flow_type'] = 'undefined'
            result_msg = f"[UNKNOWN -> UNDEFINED] {file_name}"
    else:
        warnings.warn(
            f"File '{file_name}' not listed in series config 'files' "
            f"for series '{series_cfg['name']}', setting to 'undefined'"
        )
        df['flow_type'] = 'undefined'
        result_msg = f"[UNLISTED -> UNDEFINED] {file_name}"

    if plot and img_path is not None:
        plot_series(df, title=file_name, series_name=series_cfg['name'], img_path=img_path)

    # Save to processed/<series>/<file>.csv
    save_df(df, series_cfg['name'], label_config['output_root'],
            file_name=file_name, data_path=data_path)

    gc.collect()
    return result_msg


def process_series(
    series_cfg: dict,
    label_config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1
) -> None:
    """
    Process all files in a series: load, classify, optionally plot, and save.

    Args:
        series_cfg: Series configuration dict.
        label_config: Label config from label.json.
        data_path: Base data path.
        img_path: Path for saving plots (optional).
        plot: Whether to create plots.
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel processes to use.
    """
    input_dir = data_path / label_config['input_root'] / series_cfg['input_subdir']

    csv_files = sorted(input_dir.glob("*.csv"))
    if only_files:
        csv_files = [f for f in csv_files if f.stem in only_files]

    # Read file flow types from config
    file_flow_types = series_cfg.get('files')
    if file_flow_types:
        print(f"  Config 'files' mapping: {len(file_flow_types)} files")
    else:
        warnings.warn(
            f"No 'files' mapping in config for series '{series_cfg['name']}'. "
            f"All files will be labeled as 'undefined'."
        )

    if max_workers > 1 and len(csv_files) > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_single_file, file_path, series_cfg,
                    label_config, data_path, img_path, plot, file_flow_types
                )
                for file_path in csv_files
            ]

            with tqdm(total=len(csv_files), desc=f"  {series_cfg['name']}", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    tqdm.write(f"  {result}")
                    pbar.update(1)
    else:
        # Sequential processing
        for file_path in tqdm(csv_files, desc=f"  {series_cfg['name']}", unit="file"):
            result = process_single_file(
                file_path, series_cfg, label_config, data_path,
                img_path, plot, file_flow_types
            )
            tqdm.write(f"  {result}")


def process_label(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot: bool = True,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1
) -> None:
    """
    Main entry point for label processing.

    Args:
        config: Full config dict (must contain 'label').
        data_path: Base data path.
        img_path: Path for saving plots (optional).
        plot: Whether to create plots.
        only_files: Optional list of file stems to process.
        max_workers: Number of parallel processes to use.
    """
    label_config = config['label']

    print("=" * 60)
    print("LABEL DATA PIPELINE")
    print("=" * 60)
    print(f"Input:  {label_config['input_root']}")
    print(f"Output: {label_config['output_root']}")
    print(f"Series: {[s['name'] for s in label_config['series']]}")
    print(f"Workers: {max_workers}")
    print("=" * 60)

    for series in label_config['series']:
        if series.get('processed', False):
            print(f"\n[SKIP] {series['name']} (already processed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"SERIES: {series['name']}")
        print(f"{'=' * 60}")
        process_series(series, label_config, data_path, img_path, plot, only_files=only_files, max_workers=max_workers)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
