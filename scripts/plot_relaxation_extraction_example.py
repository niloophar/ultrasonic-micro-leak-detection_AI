#!/usr/bin/env python
"""
Thesis figure — thermal-relaxation extraction from one representative
processed recording. Three vertically stacked, time-aligned panels:
(a) cleaned signal + block means, (b) block-to-block jump detection,
(c) final relaxation segmentation.

Reuses the actual pipeline function rather than re-deriving the
segmentation logic: ``add_relaxation_columns`` (src/preprocessing/
relaxation.py) is called wholesale, exactly as relaxation.py's own
``process_single_file`` calls it — same MAD-based jump detection
(``detect_jumps_mad``), same minimum-length filter, same block
aggregation (``aggregate_to_blocks``). Parameters come from
``configs/preprocessing/relaxation.json``.

NOTE ON KINK DETECTION: an earlier version of the pipeline had a
``src/preprocessing/kink.py`` step (MAD/curvature/online-prediction kink
detection within each relaxation). It was deliberately removed from
production in commit c6bc57a ("kink is no longer part of the production
preprocessing pipeline — downstream work operates purely on the
equilibrium module's output"); the module, its config, and the
Relaxation.kinks field no longer exist. This figure therefore shows
jump boundaries and final relaxation segments only — there is no kink
layer to reuse or reproduce faithfully.

Run (needs the full data/processed/<series>/*.csv tree — see CLAUDE.md's
"Development workflow"):

    python scripts/plot_relaxation_extraction_example.py

Outputs:
    figures/relaxation_extraction_example.{pdf,png}
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

from src.preprocessing.relaxation import (
    aggregate_to_blocks,
    detect_jumps_mad,
    add_relaxation_columns,
)

# =============================================================================
# STYLE (Okabe-Ito colour-blind-safe palette)
# =============================================================================

SIGNAL_COLOR = '#1a1a1a'      # near-black — cleaned point-level signal
BLOCK_COLOR = '#0072B2'       # blue — block means
JUMP_COLOR = '#D55E00'        # vermillion — jump-classified diffs / boundaries
THRESH_COLOR = '#D55E00'
BAND_COLORS = ['#4C6B8A', '#B0B0B0']  # alternating subtle segment bands
BAND_ALPHA = 0.10

LW_SIGNAL = 0.7
LW_BLOCK = 1.0
MS_BLOCK = 3.5

SHORTLIST_SIZE = 6
N_SEG_TARGET = 4          # relaxation segments to show in the focused window
MARGIN_BLOCKS = 25        # context blocks before/after the shown segments


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
    ax.text(-0.06, 1.05, text, transform=ax.transAxes,
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
# CANDIDATE SHORTLISTING (cheap prefilter) + EXACT RECOMPUTATION
# =============================================================================

def scan_relaxation_count(path: Path) -> int:
    """Cheap prefilter: number of distinct final relaxation_id values already
    stored by the pipeline (proxy for 'several valid final segments')."""
    try:
        s = pd.read_csv(path, usecols=['relaxation_id'])['relaxation_id']
    except (ValueError, pd.errors.EmptyDataError):
        return 0
    return int((s.unique() >= 0).sum())


def run_relaxation_extraction(path: Path, relaxation_config: dict) -> Optional[dict]:
    """Load one file and run add_relaxation_columns() exactly as
    relaxation.py's own pipeline step does. Returns None on load failure."""
    try:
        df = pd.read_csv(path, index_col=False)
    except (ValueError, pd.errors.EmptyDataError):
        return None
    time_col = relaxation_config['time_col']
    if time_col not in df.columns or 'valid' not in df.columns:
        return None
    df[time_col] = pd.to_datetime(df[time_col])

    df_result, df_blocks, jump_block_indices, metadata_list = add_relaxation_columns(
        df, relaxation_config)
    if df_blocks.empty:
        return None

    return {
        'df_result': df_result,
        'df_blocks': df_blocks,
        'jump_block_indices': jump_block_indices,
        'metadata_list': metadata_list,
    }


def select_example(
    data_root: Path, house: str, relaxation_config: dict,
    manual_series: Optional[str], manual_file: Optional[str],
) -> Tuple[str, Path, dict, dict]:
    """Returns (series_name, file_path, extraction_result, criteria_report)."""
    if manual_series and manual_file:
        series_dir = data_root / manual_series
        target = series_dir / manual_file
        if not target.suffix:
            target = series_dir / f'{manual_file}.csv'
        if not target.exists():
            raise RuntimeError(f'Manual file not found: {target}')
        result = run_relaxation_extraction(target, relaxation_config)
        if result is None:
            raise RuntimeError(f'{target} has no usable relaxation/valid columns.')
        report = criteria_report(result)
        return manual_series, target, result, report

    best_overall = None  # (score, series_name, file_path, result, report)
    for house_pass in house_passes(house):
        pool = []  # (n_relax_stored, series_name, file_path)
        for series_dir in ordered_series_dirs(data_root, house_pass if house_pass != 'any' else house):
            if house_pass in ('A', 'B'):
                if house_pass == 'A' and is_house_b(series_dir.name):
                    continue
                if house_pass == 'B' and not is_house_b(series_dir.name):
                    continue
            for f in sorted(series_dir.glob('*.csv')):
                n = scan_relaxation_count(f)
                if n > 0:
                    pool.append((n, series_dir.name, f))
        pool.sort(key=lambda x: x[0], reverse=True)

        for _, series_name, file_path in pool[:SHORTLIST_SIZE]:
            result = run_relaxation_extraction(file_path, relaxation_config)
            if result is None:
                continue
            report = criteria_report(result)
            score = (len(result['metadata_list']), len(result['jump_block_indices']))
            if best_overall is None or score > best_overall[0]:
                best_overall = (score, series_name, file_path, result, report)
            if report['fully_qualifies']:
                return series_name, file_path, result, report

    if best_overall is None:
        raise RuntimeError(
            f'No file with any detected relaxation segment found under {data_root}. '
            'Sync more data/processed <series> directories from the GPU server '
            '(see CLAUDE.md "Development workflow"), or select one manually with '
            '--series/--file.'
        )
    _, series_name, file_path, result, report = best_overall
    return series_name, file_path, result, report


def criteria_report(result: dict) -> dict:
    n_jumps = len(result['jump_block_indices'])
    n_segments = len(result['metadata_list'])
    has_two_jumps = n_jumps >= 2
    has_several_segments = n_segments >= 3
    return {
        'n_jumps': n_jumps,
        'n_segments': n_segments,
        'has_two_jumps': has_two_jumps,
        'has_several_segments': has_several_segments,
        'fully_qualifies': has_two_jumps and has_several_segments,
    }


# =============================================================================
# WINDOW SELECTION (focused sub-range, not the full recording)
# =============================================================================

def choose_window(metadata_list: List[dict], n_blocks_total: int) -> Tuple[int, int, List[dict]]:
    """Pick the contiguous run of N_SEG_TARGET relaxation segments that gives
    the cleanest, most 'textbook' sawtooth: longest average segment duration
    (favours clear, well-separated jump-then-decay events over a cluster of
    short, choppy segments), subject to a relative span cap so the window
    stays focused rather than sprawling across the whole recording."""
    n = len(metadata_list)
    n_show = min(N_SEG_TARGET, n)
    if n_show == 0:
        return 0, n_blocks_total, []

    max_span = max(1, int(n_blocks_total * 0.15))

    def window_key(i: int, enforce_cap: bool):
        segs = metadata_list[i:i + n_show]
        span = segs[-1]['end_block'] - segs[0]['start_block']
        avg_len = sum(s['n_blocks'] for s in segs) / n_show
        if enforce_cap and span > max_span:
            return None
        return (avg_len, -span)

    best_idx, best_key = 0, None
    for enforce_cap in (True, False):
        for i in range(0, n - n_show + 1):
            key = window_key(i, enforce_cap)
            if key is None:
                continue
            if best_key is None or key > best_key:
                best_key, best_idx = key, i
        if best_key is not None:
            break

    shown = metadata_list[best_idx:best_idx + n_show]
    win_start = max(0, shown[0]['start_block'] - MARGIN_BLOCKS)
    win_end = min(n_blocks_total, shown[-1]['end_block'] + MARGIN_BLOCKS)
    return win_start, win_end, shown


# =============================================================================
# FIGURE
# =============================================================================

def plot_figure(
    series_name: str, target: Path, result: dict, relaxation_config: dict,
    output_dir: Path, report: dict,
) -> None:
    value_col = relaxation_config['value_col']
    time_col = relaxation_config['time_col']
    jump_cfg = relaxation_config['relaxation_detection']['jump_detection']
    threshold_factor = jump_cfg.get('threshold_factor', 5.0)
    min_length_points = relaxation_config['relaxation_detection']['min_relaxation_length']
    block_size = jump_cfg.get('block_size', 64)
    min_length_blocks = max(1, min_length_points // block_size)

    df_result = result['df_result']
    df_blocks = result['df_blocks']
    jump_block_indices = result['jump_block_indices']
    metadata_list = result['metadata_list']

    block_means = df_blocks['block_mean'].values
    diffs = np.diff(block_means)
    _, threshold, robust_std = detect_jumps_mad(block_means, threshold_factor)
    median_diff = float(np.median(diffs)) if len(diffs) else 0.0
    jump_mask = np.abs(diffs - median_diff) > threshold  # same condition as detect_jumps_mad

    n_blocks_total = len(df_blocks)
    win_start, win_end, shown_segments = choose_window(metadata_list, n_blocks_total)

    block_times = pd.to_datetime(df_blocks['block_time'])
    win_time_start = block_times.iloc[win_start]
    win_time_end = block_times.iloc[min(win_end, n_blocks_total - 1)]

    # Point-level signal restricted to the displayed window (valid points only).
    df_valid = df_result[df_result['valid'] == 1].sort_values(time_col)
    df_win = df_valid[(df_valid[time_col] >= win_time_start) & (df_valid[time_col] <= win_time_end)]

    blocks_win = df_blocks.iloc[win_start:win_end]

    # Diffs indexed 0..n_blocks-2; diffs[i] is the jump from block i to i+1,
    # displayed at block i+1's time (the new block the jump introduces).
    diff_win_idx = [i for i in range(len(diffs)) if win_start <= i + 1 < win_end]
    diff_times = [block_times.iloc[i + 1] for i in diff_win_idx]
    diff_vals = [diffs[i] for i in diff_win_idx]
    diff_is_jump = [jump_mask[i] for i in diff_win_idx]

    jumps_win = [j for j in jump_block_indices if win_start <= j < win_end]
    jump_times = [block_times.iloc[j] for j in jumps_win]

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.0, 7.9), constrained_layout=True, sharex=True)
    fig.get_layout_engine().set(h_pad=0.05, hspace=0.02)
    for ax in (ax1, ax2, ax3):
        ax.margins(x=0.02)

    # --- Panel (a): cleaned signal + block means ---
    ax1.plot(df_win[time_col], df_win[value_col], color=SIGNAL_COLOR, lw=LW_SIGNAL,
              alpha=0.85, zorder=2, label='Signal')
    ax1.plot(blocks_win['block_time'], blocks_win['block_mean'], color=BLOCK_COLOR,
              lw=LW_BLOCK, marker='o', ms=MS_BLOCK, zorder=3, label='Block mean')
    ax1.set_ylabel(axis_label(value_col))
    ax1.grid(True, color='#cccccc', alpha=0.35, linewidth=0.5)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.88, fontsize=7.5,
               handlelength=1.6, borderaxespad=0.6)
    panel_label(ax1, '(a)')

    # --- Panel (b): jump detection ---
    ax2.plot(diff_times, diff_vals, color='#888888', lw=0.9, alpha=0.8, zorder=2)
    ax2.scatter(diff_times, diff_vals, color='#888888', s=10, alpha=0.8, zorder=2)
    jump_pts_t = [t for t, is_j in zip(diff_times, diff_is_jump) if is_j]
    jump_pts_v = [v for v, is_j in zip(diff_vals, diff_is_jump) if is_j]
    if jump_pts_t:
        ax2.scatter(jump_pts_t, jump_pts_v, color=JUMP_COLOR, s=32, zorder=4,
                    label='Jump-classified diff')
    ax2.axhline(median_diff + threshold, color=THRESH_COLOR, linestyle='--', lw=1.1)
    ax2.axhline(median_diff - threshold, color=THRESH_COLOR, linestyle='--', lw=1.1)
    yt = ax2.get_yaxis_transform()
    ax2.annotate(f'±thresh = {threshold:.3g}', xy=(0.985, median_diff + threshold),
                 xycoords=yt, xytext=(0, 5), textcoords='offset points',
                 color=THRESH_COLOR, fontsize=7, va='bottom', ha='right')
    for jt in jump_times:
        ax2.axvline(jt, color=JUMP_COLOR, lw=0.8, alpha=0.6, zorder=1)
    diff_span = max(diff_vals) - min(diff_vals)
    diff_pad = diff_span * 0.12 if diff_span > 0 else 1.0
    ax2.set_ylim(min(diff_vals) - diff_pad, max(diff_vals) + diff_pad)
    ax2.set_ylabel(f'{axis_label(value_col)}\n(block diff)')
    ax2.grid(True, color='#cccccc', alpha=0.35, linewidth=0.5)
    if jump_pts_t:
        ax2.legend(loc='upper left', frameon=True, framealpha=0.88, fontsize=7.5,
                   handlelength=1.6, borderaxespad=0.6)
    panel_label(ax2, '(b)')

    # --- Panel (c): final relaxation segmentation ---
    ax3.plot(df_win[time_col], df_win[value_col], color=SIGNAL_COLOR, lw=LW_SIGNAL,
              alpha=0.85, zorder=2)

    y_lo, y_hi = df_win[value_col].min(), df_win[value_col].max()
    pad = (y_hi - y_lo) * 0.08 if y_hi > y_lo else 1.0
    ax3.set_ylim(y_lo - pad, y_hi + pad)

    win_span = (win_time_end - win_time_start).total_seconds()
    prev_frac = None
    stagger = False
    for i, seg in enumerate(shown_segments):
        seg_start_block = max(seg['start_block'], win_start)
        seg_end_block = min(seg['end_block'] - 1, win_end - 1)
        t0 = block_times.iloc[seg_start_block]
        t1 = block_times.iloc[seg_end_block]
        ax3.axvspan(t0, t1, color=BAND_COLORS[i % 2], alpha=BAND_ALPHA, lw=0, zorder=1)
        mid = t0 + (t1 - t0) / 2

        frac = (mid - win_time_start).total_seconds() / win_span if win_span > 0 else 0.0
        stagger = (prev_frac is not None and (frac - prev_frac) < 0.06 and not stagger)
        prev_frac = frac
        y_off = 12 if stagger else 2
        ax3.annotate(f"R{seg['relaxation_id']}", xy=(mid, 1.0), xycoords=('data', 'axes fraction'),
                     xytext=(0, y_off), textcoords='offset points', ha='center', va='bottom',
                     fontsize=8, color='#333333')

    for jt in jump_times:
        ax3.axvline(jt, color=JUMP_COLOR, linestyle='-', lw=1.1, zorder=3)

    ax3.set_xlabel('Time')
    ax3.set_ylabel(axis_label(value_col))
    ax3.grid(True, color='#cccccc', alpha=0.35, linewidth=0.5)

    legend_c = [
        Line2D([0], [0], color=SIGNAL_COLOR, lw=1.4, label='Signal'),
        Line2D([0], [0], color=JUMP_COLOR, lw=1.4, label='Jump boundary'),
        Patch(facecolor=BAND_COLORS[0], alpha=BAND_ALPHA + 0.10, edgecolor='none', label='Relaxation segment'),
    ]
    ax3.legend(handles=legend_c, loc='lower left', frameon=True, framealpha=0.88,
               fontsize=7.5, handlelength=1.6, borderaxespad=0.6)
    format_time_axis(ax3)
    panel_label(ax3, '(c)')

    save_fig(fig, 'relaxation_extraction_example', output_dir)
    plt.close(fig)

    print(f'  series:                    {series_name}')
    print(f'  source file:               {target.name}')
    print(f'  displayed signal:          {value_col}')
    print(f'  number of blocks:          {n_blocks_total:,} (whole file)')
    print(f'  number of detected jumps:  {report["n_jumps"]} (whole file)')
    print(f'  number of detected kinks:  0 — kink detection removed from the '
          f'production pipeline (commit c6bc57a), not part of the current algorithm')
    print(f'  final relaxation segments: {report["n_segments"]} (whole file)')
    print(f'  jump threshold:            +/-{threshold:.4g} ps '
          f'(= {threshold_factor:g} x 1.4826 x MAD(diff - median_diff), median_diff={median_diff:.4g})')
    print(f'  minimum segment length:    {min_length_points} points '
          f'({min_length_blocks} block(s) at block_size={block_size})')
    print(f'  kink persistence param:    N/A — kink step removed from the pipeline')
    missing = []
    if not report['has_two_jumps']:
        missing.append('>=2 detected jumps')
    if not report['has_several_segments']:
        missing.append('>=3 final relaxation segments')
    print(f'  selection criteria:        '
          f'{"all satisfied" if not missing else "PARTIAL — missing: " + ", ".join(missing)}')


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--series', type=str, default=None,
                    help='Manual series override (directory name under --data-root). '
                         'Must be given together with --file.')
    p.add_argument('--file', type=str, default=None,
                    help='Manual file override (stem or filename, requires --series).')
    p.add_argument('--data-root', type=str, default='data/processed',
                    help='Root directory of processed series (default: data/processed).')
    p.add_argument('--config-dir', type=str, default='configs/preprocessing',
                    help='Directory with relaxation.json.')
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
    if not args.data_root.exists():
        raise RuntimeError(f'--data-root {args.data_root} does not exist.')

    with open(args.config_dir / 'relaxation.json') as f:
        relaxation_config = json.load(f)

    print('\n=== Relaxation extraction example ===')
    series_name, target, result, report = select_example(
        args.data_root, args.house, relaxation_config, args.series, args.file)

    plot_figure(series_name, target, result, relaxation_config, args.output_dir, report)


if __name__ == '__main__':
    main()
