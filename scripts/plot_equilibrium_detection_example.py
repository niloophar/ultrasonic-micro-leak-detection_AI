#!/usr/bin/env python
"""
Thesis figure — equilibrium detection within one representative thermal
relaxation segment. Two vertically stacked, time-aligned panels: (a) the
raw signal with the detected equilibrium interval shaded, (b) the
sliding-window slope estimate with the +/-tau_eq thresholds and the same
shaded interval.

Reuses the actual pipeline functions rather than re-deriving the
algorithm:
  - ``aggregate_to_blocks`` (src/preprocessing/relaxation.py) — point -> block
    aggregation, block_size from equilibrium.json.
  - ``detect_equilibrium_regions`` (src/preprocessing/equilibrium.py) — the
    sliding-window linear-regression slope estimate, +/-tau_eq thresholding,
    gap bridging, and consecutive-window merging. Called with the exact
    parameters from ``configs/preprocessing/equilibrium.json``.
  - ``get_leakage_time_windows`` / ``relaxation_overlaps_leakage`` — the
    same leakage-overlap exclusion the pipeline's non-leakage equilibrium
    pass applies, so the selected relaxation is guaranteed to be one the
    pipeline actually ran non-leakage detection on.

The ``relaxation_id`` column (written by relaxation.py, step 3) and the
``equilibrium`` column (written by equilibrium.py, step 5) are read directly
from ``data/processed/<series>/<file>.csv`` to cheaply shortlist a good
example before the exact sliding-window recomputation runs on the winning
candidate only — this script never invents thresholds or overrides the
pipeline's own detection outcome.

Run (needs the full data/processed/<series>/*.csv tree — see CLAUDE.md's
"Development workflow"):

    python scripts/plot_equilibrium_detection_example.py

Outputs:
    figures/equilibrium_detection_example.{pdf,png}
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.preprocessing.relaxation import aggregate_to_blocks
from src.preprocessing.equilibrium import (
    detect_equilibrium_regions,
    get_leakage_time_windows,
    relaxation_overlaps_leakage,
)

# =============================================================================
# STYLE
# =============================================================================

SIGNAL_COLOR = '#333333'   # dark grey — full (transient) signal
EQ_COLOR = '#1f5fa8'       # blue — equilibrium portion / stable windows
BAND_COLOR = '#1f5fa8'     # shaded equilibrium band (same hue, low alpha)
THRESH_COLOR = '#d62728'   # red — +/-tau_eq threshold lines

LW_SIGNAL = 0.7
LW_EQ_OVERLAY = 1.1
BAND_ALPHA = 0.15

# Cheap-prefilter shortlist size: try the top-N stored-equilibrium candidates
# (by score) through the exact recomputation before falling back to the
# other house, so a rare leakage-overlap mismatch doesn't abort the search.
SHORTLIST_SIZE = 8

MIN_TRANSIENT_POINTS_PREFERRED = 20 * 64  # ~20 blocks of visible transient


def apply_style() -> None:
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10.5,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#1a1a1a',
        'text.color': '#1a1a1a',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def save_fig(fig, name: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [('pdf', None), ('png', 300)]:
        path = figures_dir / f'{name}.{ext}'
        kw = dict(bbox_inches='tight', facecolor='white')
        if dpi:
            kw['dpi'] = dpi
        fig.savefig(path, **kw)
        print(f'  Saved: {path}')


def axis_label(col: str) -> str:
    if ' in ' in col:
        name, unit = col.split(' in ', 1)
        return f'{name} ({unit})'
    return col


def format_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def panel_label(ax, text: str) -> None:
    ax.text(-0.06, 1.03, text, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='right')


def is_house_b(series_name: str) -> bool:
    return series_name.startswith('hb-')


def ordered_series_dirs(data_root: Path, house: str) -> List[Path]:
    all_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    house_a = [p for p in all_dirs if not is_house_b(p.name)]
    house_b = [p for p in all_dirs if is_house_b(p.name)]
    if house == 'A':
        return house_a + house_b
    elif house == 'B':
        return house_b + house_a
    return all_dirs


def house_passes(house: str) -> List[str]:
    if house == 'A':
        return ['A', 'B']
    if house == 'B':
        return ['B', 'A']
    return ['any']


# =============================================================================
# CANDIDATE SHORTLISTING (cheap prefilter on stored columns)
# =============================================================================

def scan_file_candidates(path: Path, min_total_points: int) -> List[dict]:
    """Per-relaxation summary from the columns the pipeline already wrote,
    used only to rank candidates before the exact recomputation."""
    try:
        df = pd.read_csv(path, usecols=['relaxation_id', 'valid', 'equilibrium'])
    except (ValueError, pd.errors.EmptyDataError):
        return []
    if not {'relaxation_id', 'valid', 'equilibrium'}.issubset(df.columns):
        return []

    out = []
    rel_ids = sorted(r for r in df['relaxation_id'].unique() if r >= 0)
    for rid in rel_ids:
        rel_valid = df.loc[(df['relaxation_id'] == rid) & (df['valid'] == 1), 'equilibrium']
        n_total = len(rel_valid)
        if n_total < min_total_points:
            continue
        eq_positions = np.where(rel_valid.to_numpy() == 1)[0]
        if len(eq_positions) == 0:
            continue
        out.append({
            'rel_id': int(rid),
            'n_total': n_total,
            'n_eq': int(len(eq_positions)),
            'first_eq': int(eq_positions[0]),
            'last_eq': int(eq_positions[-1]),
        })
    return out


def candidate_score(c: dict) -> float:
    """Reward a substantial equilibrium interval AND a visible transient
    before it, so the figure shows both a clear transient and a clear
    stable tail rather than a relaxation that is equilibrium almost
    everywhere."""
    return min(c['n_eq'], 20000) + 0.5 * min(c['first_eq'], 20000)


# =============================================================================
# EXACT RECOMPUTATION (same functions/config the pipeline uses)
# =============================================================================

def recompute_relaxation_equilibrium(
    df: pd.DataFrame, rel_id: int, eq_config: dict, label_config: dict, series_name: str,
) -> Optional[dict]:
    """Reproduce equilibrium.py's non-leakage detection path exactly for one
    relaxation: filter valid points, skip if it overlaps a leakage window,
    aggregate to blocks, run detect_equilibrium_regions. Returns None if this
    relaxation doesn't qualify (mirrors the pipeline's own skip conditions)."""
    value_col = eq_config['value_col']
    time_col = eq_config['time_col']
    block_size = eq_config.get('block_size', 64)

    rel_mask = (df['relaxation_id'] == rel_id) & (df['valid'] == 1)
    df_rel = df[rel_mask].sort_values(time_col).reset_index(drop=True)
    if len(df_rel) < block_size:
        return None

    leakage_windows = get_leakage_time_windows(series_name, label_config, df=df, time_col=time_col)
    rel_start_time = pd.Timestamp(df_rel[time_col].min())
    rel_end_time = pd.Timestamp(df_rel[time_col].max())
    if leakage_windows and relaxation_overlaps_leakage(rel_start_time, rel_end_time, leakage_windows):
        return None

    df_blocks = aggregate_to_blocks(df_rel, value_col, time_col, block_size)
    if len(df_blocks) == 0:
        return None
    df_blocks['block_time'] = pd.to_datetime(df_blocks['block_time'])

    max_rmse = eq_config.get('max_rmse') if eq_config.get('residuals_filter_enabled', True) else None
    segments, eq_mask, slopes = detect_equilibrium_regions(
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
    if not segments:
        return None

    # Longest merged equilibrium segment for this relaxation.
    segment = max(segments, key=lambda s: s['n_blocks'])

    return {
        'df_rel': df_rel,
        'df_blocks': df_blocks,
        'eq_mask': eq_mask,
        'slopes': slopes,
        'segment': segment,
        'value_col': value_col,
        'time_col': time_col,
        'block_size': block_size,
    }


def select_example(
    data_root: Path, house: str, eq_config: dict, label_config: dict,
    manual_series: Optional[str], manual_file: Optional[str], manual_rel_id: Optional[int],
) -> Tuple[str, Path, int, dict]:
    block_size = eq_config.get('block_size', 64)
    window_length = eq_config.get('window_length', 250)
    min_total_points = window_length * block_size

    if manual_series and manual_file:
        series_dir = data_root / manual_series
        target = series_dir / manual_file
        if not target.suffix:
            target = series_dir / f'{manual_file}.csv'
        if not target.exists():
            raise RuntimeError(f'Manual file not found: {target}')
        df = pd.read_csv(target)
        df[eq_config['time_col']] = pd.to_datetime(df[eq_config['time_col']])
        if manual_rel_id is not None:
            result = recompute_relaxation_equilibrium(
                df, manual_rel_id, eq_config, label_config, manual_series)
            if result is None:
                raise RuntimeError(
                    f'relaxation_id={manual_rel_id} in {target} has no qualifying '
                    'equilibrium segment (too short, or overlaps a leakage window).')
            return manual_series, target, manual_rel_id, result
        # No relaxation id given: pick the best-scoring one in this file.
        candidates = scan_file_candidates(target, min_total_points)
        if not candidates:
            raise RuntimeError(f'No relaxation with a detected equilibrium interval in {target}.')
        candidates.sort(key=candidate_score, reverse=True)
        for c in candidates[:SHORTLIST_SIZE]:
            result = recompute_relaxation_equilibrium(
                df, c['rel_id'], eq_config, label_config, manual_series)
            if result is not None:
                return manual_series, target, c['rel_id'], result
        raise RuntimeError(f'No relaxation in {target} reproduced a qualifying equilibrium segment.')

    for house_pass in house_passes(house):
        pool = []  # (score, series_name, file_path, rel_id)
        candidates_by_key = ordered_series_dirs(data_root, house_pass if house_pass != 'any' else house)
        for series_dir in candidates_by_key:
            if house_pass in ('A', 'B'):
                if house_pass == 'A' and is_house_b(series_dir.name):
                    continue
                if house_pass == 'B' and not is_house_b(series_dir.name):
                    continue
            for f in sorted(series_dir.glob('*.csv')):
                for c in scan_file_candidates(f, min_total_points):
                    pool.append((candidate_score(c), series_dir.name, f, c['rel_id']))
        if not pool:
            continue
        pool.sort(key=lambda x: x[0], reverse=True)
        loaded_files = {}
        for _, series_name, file_path, rel_id in pool[:SHORTLIST_SIZE]:
            if file_path not in loaded_files:
                df = pd.read_csv(file_path)
                df[eq_config['time_col']] = pd.to_datetime(df[eq_config['time_col']])
                loaded_files[file_path] = df
            df = loaded_files[file_path]
            result = recompute_relaxation_equilibrium(df, rel_id, eq_config, label_config, series_name)
            if result is not None:
                return series_name, file_path, rel_id, result

    raise RuntimeError(
        'No relaxation segment with a qualifying, reproducible equilibrium interval '
        f'found under {data_root}. Sync more data/processed <series> directories from '
        'the GPU server (see CLAUDE.md "Development workflow"), or select one manually '
        'with --series/--file/--relaxation-id.'
    )


# =============================================================================
# FIGURE
# =============================================================================

def plot_figure(series_name: str, target: Path, rel_id: int, result: dict, eq_config: dict,
                 output_dir: Path) -> None:
    df_rel = result['df_rel']
    df_blocks = result['df_blocks']
    eq_mask = result['eq_mask']
    slopes = result['slopes']
    segment = result['segment']
    value_col = result['value_col']
    time_col = result['time_col']
    block_size = result['block_size']

    window_length = eq_config.get('window_length', 250)
    hop_size = eq_config.get('hop_size', 50)
    slope_threshold = eq_config.get('slope_threshold', 0.75)

    start_block, end_block = segment['start_block'], segment['end_block']
    eq_start_pt = int(df_blocks.iloc[start_block]['start_idx'])
    eq_end_pt = int(df_blocks.iloc[end_block - 1]['end_idx']) - 1
    eq_time_start = df_rel[time_col].iloc[eq_start_pt]
    eq_time_end = df_rel[time_col].iloc[eq_end_pt]

    # Window center times, matching the pipeline's own diagnostic plot
    # (src/common/plotting.py::plot_equilibrium_detection).
    window_starts = list(range(0, len(df_blocks) - window_length + 1, hop_size))
    block_times = pd.to_datetime(df_blocks['block_time'])
    window_centers = [
        block_times.iloc[min(ws + window_length // 2, len(block_times) - 1)]
        for ws in window_starts
    ]
    window_is_eq = np.abs(slopes) < slope_threshold
    n_stable_windows = int(window_is_eq.sum())

    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.6), constrained_layout=True, sharex=True)

    # --- Panel (a): relaxation signal ---
    ax1.plot(df_rel[time_col], df_rel[value_col], color=SIGNAL_COLOR, lw=LW_SIGNAL,
              alpha=0.9, rasterized=True, zorder=2)
    eq_pt_mask = (df_rel[time_col] >= eq_time_start) & (df_rel[time_col] <= eq_time_end)
    ax1.plot(df_rel.loc[eq_pt_mask, time_col], df_rel.loc[eq_pt_mask, value_col],
              color=EQ_COLOR, lw=LW_EQ_OVERLAY, alpha=0.95, rasterized=True, zorder=3)
    ax1.axvspan(eq_time_start, eq_time_end, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=1)

    ax1.set_ylabel(axis_label(value_col))
    ax1.grid(True, color='#cccccc', alpha=0.35, linewidth=0.5)
    legend_a = [
        Line2D([0], [0], color=SIGNAL_COLOR, lw=1.4, label='Transient signal'),
        Line2D([0], [0], color=EQ_COLOR, lw=1.4, label='Equilibrium interval'),
        Patch(facecolor=BAND_COLOR, alpha=BAND_ALPHA, edgecolor='none', label='Equilibrium (merged)'),
    ]
    ax1.legend(handles=legend_a, loc='upper right', frameon=True, framealpha=0.88,
               fontsize=7.5, handlelength=1.6, borderaxespad=0.6)
    panel_label(ax1, '(a)')

    # --- Panel (b): local slope estimate ---
    ax2.plot(window_centers, slopes, color=SIGNAL_COLOR, lw=1.0, alpha=0.85, zorder=2,
             label='Window slope')
    if n_stable_windows:
        stable_times = [t for t, is_eq in zip(window_centers, window_is_eq) if is_eq]
        stable_slopes = slopes[window_is_eq]
        ax2.scatter(stable_times, stable_slopes, color=EQ_COLOR, s=16, zorder=4,
                    label='Stable window')
    ax2.axhline(slope_threshold, color=THRESH_COLOR, linestyle='--', lw=1.1)
    ax2.axhline(-slope_threshold, color=THRESH_COLOR, linestyle='--', lw=1.1)
    ax2.text(0.985, slope_threshold, f' +τ_eq={slope_threshold:g}', transform=ax2.get_yaxis_transform(),
              color=THRESH_COLOR, fontsize=7, va='bottom', ha='right')
    ax2.text(0.985, -slope_threshold, f' -τ_eq={slope_threshold:g}', transform=ax2.get_yaxis_transform(),
              color=THRESH_COLOR, fontsize=7, va='top', ha='right')
    ax2.axvspan(eq_time_start, eq_time_end, color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=1)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Slope (ps/s)')
    ax2.grid(True, color='#cccccc', alpha=0.35, linewidth=0.5)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ['Window slope', 'Stable window']
    ax2.legend([by_label[k] for k in order if k in by_label], [k for k in order if k in by_label],
               loc='lower right', frameon=True, framealpha=0.88, fontsize=7.5,
               handlelength=1.6, borderaxespad=0.6)
    format_time_axis(ax2)
    panel_label(ax2, '(b)')

    save_fig(fig, 'equilibrium_detection_example', output_dir)
    plt.close(fig)

    duration_min = segment['duration_seconds'] / 60.0
    print(f'  series:                    {series_name}')
    print(f'  source file:               {target.name}')
    print(f'  displayed signal:          {value_col}')
    print(f'  relaxation identifier:     {rel_id}')
    print(f'  equilibrium slope thresh:  +/-{slope_threshold:g} ps/s')
    print(f'  window length:             {window_length} blocks ({window_length * block_size:,} points)')
    print(f'  number of stable windows:  {n_stable_windows} / {len(slopes)}')
    print(f'  final equilibrium interval: {segment["n_blocks"]} blocks, '
          f'{duration_min:.1f} min ({segment["duration_seconds"]:.0f} s)')


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--series', type=str, default=None,
                    help='Manual series override (directory name under --data-root). '
                         'Must be given together with --file.')
    p.add_argument('--file', type=str, default=None,
                    help='Manual file override (stem or filename, requires --series).')
    p.add_argument('--relaxation-id', type=int, default=None,
                    help='Manual relaxation_id override (requires --series/--file). '
                         'If omitted with --series/--file, the best-scoring relaxation '
                         'in that file is chosen automatically.')
    p.add_argument('--data-root', type=str, default='data/processed',
                    help='Root directory of processed series (default: data/processed).')
    p.add_argument('--config-dir', type=str, default='configs/preprocessing',
                    help='Directory with equilibrium.json / label.json.')
    p.add_argument('--output-dir', type=str, default='figures',
                    help='Output directory for the figure (default: figures).')
    p.add_argument('--house', choices=['A', 'B', 'any'], default='A',
                    help='House preference for automatic selection (default: A). '
                         'Falls back to the other house if none qualifies.')
    return p


def main():
    args = build_arg_parser().parse_args()
    args.data_root = Path(args.data_root)
    args.config_dir = Path(args.config_dir)
    args.output_dir = Path(args.output_dir)

    if bool(args.series) != bool(args.file):
        raise RuntimeError('--series and --file must be given together.')
    if args.relaxation_id is not None and not args.series:
        raise RuntimeError('--relaxation-id requires --series/--file.')
    if not args.data_root.exists():
        raise RuntimeError(f'--data-root {args.data_root} does not exist.')

    with open(args.config_dir / 'equilibrium.json') as f:
        eq_config = json.load(f)
    with open(args.config_dir / 'label.json') as f:
        label_config = json.load(f)

    print('\n=== Equilibrium detection example ===')
    series_name, target, rel_id, result = select_example(
        args.data_root, args.house, eq_config, label_config,
        args.series, args.file, args.relaxation_id)

    plot_figure(series_name, target, rel_id, result, eq_config, args.output_dir)


if __name__ == '__main__':
    main()
