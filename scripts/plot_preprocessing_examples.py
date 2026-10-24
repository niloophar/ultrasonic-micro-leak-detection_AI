#!/usr/bin/env python
"""
Thesis figures — Data Preprocessing chapter: labelling, global-outlier
detection, and local-outlier detection / cleaning.

Reproduces the *actual* pipeline implementation and configuration —
no synthetic data, no invented thresholds:
  - Figure 1 (labelling) uses the ``flow_type`` column already computed
    by src/preprocessing/label.py (classify_flow_types) at pipeline time.
  - Figures 2+3 (cleaning) call straight into src/preprocessing/clean.py
    (robust_zscore, compute_local_outlier_blocks) to re-derive the exact
    same Z-scores / block statistics for display, rather than
    reimplementing the formulas.
  - All thresholds come from configs/preprocessing/clean.json.

Data source: reads existing files under ``data/processed/<series>/*.csv``
(the actual pipeline output — Time, flow_type, valid columns already
computed by ``python -m src.preprocessing.main --run label clean``).
The script never writes to data/processed and never invents data.

Run (needs the full data/processed/<series>/*.csv tree — typically on
the GPU server or after syncing the relevant series down, see
CLAUDE.md's "Development workflow"):

    python scripts/plot_preprocessing_examples.py --plot all

Outputs:
    figures/preprocessing_labelling_example.{pdf,png}
    figures/global_outlier_detection.{pdf,png}
    figures/local_outlier_cleaning.{pdf,png}
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.preprocessing.clean import (
    robust_zscore,
    robust_mad_scale,
    compute_local_outlier_blocks,
)

# =============================================================================
# STYLE (shared across all three figures)
# =============================================================================

# Okabe-Ito colour-blind-safe palette
FLOW_TYPE_COLORS = {
    'noflow': '#56B4E9',       # sky blue
    'preleakage': '#E69F00',   # orange
    'flow': '#D55E00',         # vermillion
    'postleakage': '#CC79A7',  # reddish purple
    'undefined': '#999999',    # neutral grey
}
FLOW_TYPE_LABELS = {
    'noflow': 'NoFlow',
    'preleakage': 'PreLeakage',
    'flow': 'Leakage',
    'postleakage': 'PostLeakage',
    'undefined': 'Undefined',
}
FLOW_TYPE_ORDER = ['noflow', 'preleakage', 'flow', 'postleakage', 'undefined']

VALID_COLORS = {1: '#0072B2', 0: '#E69F00', -1: '#D55E00'}

SIGNAL_COLOR = '#1a1a1a'
MUTED_SIGNAL_COLOR = '#4C6B8A'  # restrained blue-grey — fig 2/3 signal + reference lines

LW_SIGNAL = 0.8
MS_POINT = 3

DEFAULT_FOCUS_HOURS_LABELLING = 3.0
DEFAULT_MAX_BLOCKS = 20


def apply_style() -> None:
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
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
    """'t_up_0 in ps' -> 't_up_0 (ps)'; passes through columns without ' in '."""
    if ' in ' in col:
        name, unit = col.split(' in ', 1)
        return f'{name} ({unit})'
    return col


def panel_label(ax, text: str) -> None:
    ax.text(-0.06, 1.02, text, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='right')


def format_time_axis(ax) -> None:
    """Adaptive tick resolution (seconds..days) so labels stay distinguishable
    regardless of the displayed window's span — a fixed HH:MM format repeats
    identical labels on the sub-minute windows Figure 3 can select."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), fontsize=8)


# =============================================================================
# CONFIG / DATA HELPERS
# =============================================================================

def load_clean_config(config_dir: Path) -> dict:
    with open(config_dir / 'clean.json') as f:
        return json.load(f)


def is_house_b(series_name: str) -> bool:
    return series_name.startswith('hb-')


def ordered_series_dirs(data_root: Path, house: str) -> List[Path]:
    """Series directories under data_root, ordered per the --house preference."""
    all_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    house_a = [p for p in all_dirs if not is_house_b(p.name)]
    house_b = [p for p in all_dirs if is_house_b(p.name)]
    if house == 'A':
        return house_a + house_b
    elif house == 'B':
        return house_b + house_a
    return all_dirs


def scan_column(path: Path, col: str) -> Optional[pd.Series]:
    try:
        return pd.read_csv(path, usecols=[col])[col]
    except (ValueError, pd.errors.EmptyDataError):
        return None


def load_full(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df.sort_values('Time').reset_index(drop=True)


def crop(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df['Time'] >= start) & (df['Time'] <= end)].reset_index(drop=True)


def block_average(df: pd.DataFrame, value_col: str, n: int = 64) -> pd.DataFrame:
    """Block-average a time-sorted frame in blocks of n rows (display smoothing only,
    same convention as plot_series() in src/common/plotting.py)."""
    d = df.reset_index(drop=True).copy()
    d['_blk'] = d.index // n
    return d.groupby('_blk', as_index=False).agg(Time=('Time', 'mean'),
                                                   **{value_col: (value_col, 'mean')})


def shade_flow_types(ax, df: pd.DataFrame) -> Dict[str, Patch]:
    """axvspan background per contiguous flow_type run + vertical boundary lines.
    Returns the legend handles for flow types actually present."""
    seg_id = (df['flow_type'] != df['flow_type'].shift()).cumsum()
    present = set()
    first = True
    for _, seg in df.groupby(seg_id):
        ft = seg['flow_type'].iloc[0]
        color = FLOW_TYPE_COLORS.get(ft, '#cccccc')
        t0, t1 = seg['Time'].iloc[0], seg['Time'].iloc[-1]
        ax.axvspan(t0, t1, color=color, alpha=0.18, lw=0)
        if not first:
            ax.axvline(t0, color='#555555', linestyle='--', linewidth=0.8, alpha=0.7)
        first = False
        present.add(ft)
    handles = {ft: Patch(facecolor=FLOW_TYPE_COLORS[ft], alpha=0.35, edgecolor='none',
                          label=FLOW_TYPE_LABELS[ft])
               for ft in FLOW_TYPE_ORDER if ft in present}
    return handles


def print_repro(header: str, info: List[Tuple[str, object]]) -> None:
    print(f'\n--- {header} ---')
    for k, v in info:
        print(f'  {k}: {v}')


def sci_y_axis(ax) -> None:
    """Force consistent scientific notation (offset exponent) on the y-axis
    instead of long decimal labels."""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)


def downsample_for_line(df: pd.DataFrame, value_col: str, target_points: int = 400) -> pd.DataFrame:
    """Block-average purely for display so a dense raw trace doesn't render
    as tens of thousands of points/markers. Detection results are computed
    beforehand on the untouched data (see robust_zscore / compute_local_outlier_blocks
    calls) and are never affected by this — it only thins what gets drawn."""
    n = len(df)
    if n <= target_points:
        return df[['Time', value_col]].reset_index(drop=True)
    factor = max(1, n // target_points)
    return block_average(df, value_col, n=factor)


def resolve_manual_window(
    args, file_start: pd.Timestamp, file_end: pd.Timestamp
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if args.window_start and args.window_end:
        ws, we = pd.Timestamp(args.window_start), pd.Timestamp(args.window_end)
        return max(file_start, ws), min(file_end, we)
    if bool(args.window_start) != bool(args.window_end):
        raise RuntimeError('--window-start and --window-end must be given together.')
    return None


def _find_isolated_run(mask: np.ndarray, min_len: int, max_len: int) -> Optional[Tuple[int, int]]:
    """
    Index range (inclusive) of the True-run in `mask` that maximises
    isolation (index-distance to its neighbouring runs) among runs whose
    length falls within [min_len, max_len]. Falls back to the best-isolated
    run of any size if none fit the bounds. Returns None if mask is all False.
    """
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    if not runs:
        return None

    sized = [k for k, (s, e) in enumerate(runs) if min_len <= (e - s + 1) <= max_len]
    pool = sized if sized else list(range(len(runs)))

    best = None
    for k in pool:
        s, e = runs[k]
        gap_before = (s - runs[k - 1][1]) if k > 0 else s
        gap_after = (runs[k + 1][0] - e) if k < len(runs) - 1 else (n - 1 - e)
        isolation = min(gap_before, gap_after)
        if best is None or isolation > best[0]:
            best = (isolation, s, e)
    return best[1], best[2]


# =============================================================================
# FIGURE 1 — LABELLING
# =============================================================================

def _score_labelling_candidate(files_ft: List[Tuple[Path, set]], i: int) -> int:
    """Score a leakage file (index i) by how much labelling context is available."""
    ft = files_ft[i][1]
    if 'flow' not in ft:
        return -1
    score = 0
    if 'preleakage' in ft and 'postleakage' in ft:
        score += 1
    prev_pure_noflow = i > 0 and files_ft[i - 1][1] == {'noflow'}
    next_pure_noflow = (i < len(files_ft) - 1) and files_ft[i + 1][1] == {'noflow'}
    score += int(prev_pure_noflow) + int(next_pure_noflow)
    return score


def select_labelling_example(
    data_root: Path, house: str, manual_series: Optional[str], manual_file: Optional[str]
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    if manual_series and manual_file:
        series_dir = data_root / manual_series
        target = series_dir / manual_file
        if not target.suffix:
            target = series_dir / f'{manual_file}.csv'
        return series_dir, target, None, None

    for house_pass in (['A', 'B'] if house == 'A' else ['B', 'A'] if house == 'B' else ['any']):
        candidates = ordered_series_dirs(data_root, house_pass if house_pass != 'any' else house)
        best = None  # (score, series_dir, target, prev, next)
        for series_dir in candidates:
            if house_pass in ('A', 'B'):
                if house_pass == 'A' and is_house_b(series_dir.name):
                    continue
                if house_pass == 'B' and not is_house_b(series_dir.name):
                    continue
            files = sorted(series_dir.glob('*.csv'))
            if not files:
                continue
            files_ft = []
            for f in files:
                s = scan_column(f, 'flow_type')
                files_ft.append((f, set(s.unique()) if s is not None else set()))
            for i in range(len(files_ft)):
                score = _score_labelling_candidate(files_ft, i)
                if score < 0:
                    continue
                if best is None or score > best[0]:
                    prev_f = files_ft[i - 1][0] if i > 0 and files_ft[i - 1][1] == {'noflow'} else None
                    next_f = files_ft[i + 1][0] if i < len(files_ft) - 1 and files_ft[i + 1][1] == {'noflow'} else None
                    best = (score, series_dir, files_ft[i][0], prev_f, next_f)
        if best is not None:
            _, series_dir, target, prev_f, next_f = best
            return series_dir, target, prev_f, next_f

    raise RuntimeError(
        'No file with flow_type=="flow" found under any series in '
        f'{data_root}. A House A leakage recording is required for the '
        'labelling figure (see CLAUDE.md "Development workflow" — sync '
        'data/processed/<series> from the GPU server, or pick a series '
        'manually with --series/--file).'
    )


def plot_labelling_figure(args) -> None:
    print('\n=== Figure 1: Labelling example ===')
    series_dir, target, prev_f, next_f = select_labelling_example(
        args.data_root, args.house, args.series, args.file)

    prev_df = load_full(prev_f) if prev_f is not None else None
    next_df = load_full(next_f) if next_f is not None else None
    parts = [load_full(target)]
    if prev_df is not None:
        parts.insert(0, prev_df)
    if next_df is not None:
        parts.append(next_df)
    df = pd.concat(parts, ignore_index=True).sort_values('Time').reset_index(drop=True)

    value_col = args.value_col
    flow_mask = df['flow_type'] == 'flow'
    flow_start, flow_end = df.loc[flow_mask, 'Time'].min(), df.loc[flow_mask, 'Time'].max()
    pad = pd.Timedelta(hours=args.focus_hours)
    # Anchor onto the neighbour file's boundary (when stitched in) rather than
    # a fixed pad from the flow window alone — pre/postleakage regions can
    # span many hours within a single day-file, so a pad measured from the
    # flow boundary alone would never reach into the neighbour's true NoFlow
    # region. Anchoring at "pad hours past the neighbour's edge" guarantees
    # NoFlow is visible whenever a neighbour was available, while still
    # capping the pre/postleakage tail when no neighbour exists.
    win_start = (prev_df['Time'].max() - pad) if prev_df is not None else (flow_start - pad)
    win_end = (next_df['Time'].min() + pad) if next_df is not None else (flow_end + pad)
    win_start = max(df['Time'].min(), win_start)
    win_end = min(df['Time'].max(), win_end)
    df_win = crop(df, win_start, win_end)

    df_blk = block_average(df_win, value_col, n=64)
    # re-attach flow_type per block (nearest original row) for shading
    df_win_sorted = df_win.sort_values('Time').reset_index(drop=True)
    df_win_sorted['_blk'] = df_win_sorted.index // 64
    blk_ft = df_win_sorted.groupby('_blk')['flow_type'].agg(lambda s: s.mode().iat[0])
    df_blk = df_blk.merge(blk_ft.rename('flow_type'), left_on='_blk', right_index=True)

    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    handles = shade_flow_types(ax, df_blk)
    ax.plot(df_blk['Time'], df_blk[value_col], color=SIGNAL_COLOR,
            lw=LW_SIGNAL, marker='.', ms=MS_POINT, alpha=0.85)

    ax.set_xlabel('Time')
    ax.set_ylabel(axis_label(value_col))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    format_time_axis(ax)
    ax.legend(handles=list(handles.values()), loc='upper left',
              bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0.0)

    save_fig(fig, 'preprocessing_labelling_example', args.figures_dir)
    plt.close(fig)

    counts = df_win['flow_type'].value_counts().to_dict()
    print_repro('Figure 1 reproducibility info', [
        ('series', series_dir.name),
        ('filename', target.stem),
        ('context files', [p.stem for p in (prev_f, next_f) if p is not None]),
        ('displayed feature', value_col),
        ('time range (displayed window)', f'{win_start} -> {win_end}'),
        ('samples per label state (in window)', counts),
        ('focus_hours (padding around leak window)', args.focus_hours),
    ])


# =============================================================================
# FIGURE 2 — GLOBAL OUTLIERS
# =============================================================================

def select_outlier_example(
    data_root: Path, house: str, manual_series: Optional[str], manual_file: Optional[str],
    kind: str
) -> Tuple[Path, Path]:
    if manual_series and manual_file:
        series_dir = data_root / manual_series
        target = series_dir / manual_file
        if not target.suffix:
            target = series_dir / f'{manual_file}.csv'
        return series_dir, target

    best = None  # (count, series_dir, file)
    for house_pass in (['A', 'B'] if house == 'A' else ['B', 'A'] if house == 'B' else ['any']):
        candidates = ordered_series_dirs(data_root, house_pass if house_pass != 'any' else house)
        found_in_pass = False
        for series_dir in candidates:
            if house_pass in ('A', 'B'):
                if house_pass == 'A' and is_house_b(series_dir.name):
                    continue
                if house_pass == 'B' and not is_house_b(series_dir.name):
                    continue
            for f in sorted(series_dir.glob('*.csv')):
                s = scan_column(f, 'valid')
                if s is None:
                    continue
                if kind == 'global':
                    count = int((s == -1).sum())
                else:
                    count = int((s == 0).sum())
                if count > 0:
                    found_in_pass = True
                if best is None or count > best[0]:
                    best = (count, series_dir, f)
        if found_in_pass:
            break
    if best is None or best[0] == 0:
        raise RuntimeError(
            f'No file with {kind} outliers (valid=={-1 if kind == "global" else 0}) '
            f'found under {data_root}. Use --series/--file to pick manually, or sync '
            'more processed series from the GPU server.'
        )
    if kind == 'global' and best[0] < 3:
        print(f'  [warning] best global-outlier candidate has only {best[0]} point(s) — '
              f'weak example. Sync more House A series for a clearer illustration.')
    return best[1], best[2]


def compute_adaptive_pad_minutes(df: pd.DataFrame) -> float:
    """Local-window padding: 20 min at high sampling rate (plenty of points
    already), up to 60 min at low sampling rate (need more time to gather
    enough points to see a trend)."""
    dt = df['Time'].diff().dropna()
    if dt.empty:
        return 30.0
    dt_s = dt.median().total_seconds()
    if dt_s <= 0.1:
        return 20.0
    if dt_s >= 2.0:
        return 60.0
    return float(20.0 + (dt_s - 0.1) / (2.0 - 0.1) * 40.0)


def select_representative_outlier(
    times: pd.Series, zvals: pd.Series, file_start: pd.Timestamp, file_end: pd.Timestamp
) -> pd.Timestamp:
    """Among candidate global-outlier timestamps, prefer an isolated one
    (large gap to its neighbours) over a member of a dense cluster; among the
    more-isolated half, prefer the largest |Z|."""
    t = (times - file_start).dt.total_seconds().to_numpy()
    fe = (file_end - file_start).total_seconds()
    n = len(t)
    if n == 1:
        return times.iloc[0]
    gap_before = np.empty(n)
    gap_after = np.empty(n)
    for i in range(n):
        gap_before[i] = t[i] - t[i - 1] if i > 0 else t[i]
        gap_after[i] = t[i + 1] - t[i] if i < n - 1 else (fe - t[i])
    isolation = np.minimum(gap_before, gap_after)
    candidates = np.where(isolation >= np.median(isolation))[0]
    if len(candidates) == 0:
        candidates = np.arange(n)
    z_abs = np.abs(zvals.to_numpy())
    best_i = candidates[np.argmax(z_abs[candidates])]
    return times.iloc[best_i]


def plot_global_outliers_figure(args, clean_cfg: dict) -> None:
    print('\n=== Figure 2: Global outlier detection ===')
    series_dir, target = select_outlier_example(
        args.data_root, args.house, args.series, args.file, kind='global')

    df = load_full(target)
    value_col = args.value_col
    global_cfg = clean_cfg['global_outlier']
    z_low, z_high = global_cfg['z_low'], global_cfg['z_high']

    # Robust Z-score computed on the FULL file's raw value_col — identical
    # subset/formula as label_outliers() in src/preprocessing/clean.py. This
    # is the only thing computed here; everything below only changes how it
    # is *displayed* (window, downsampling, insets).
    z = robust_zscore(df[value_col])
    df = df.assign(_z=z)
    z_outlier_mask = (z < z_low) | ((z > z_high) if z_high is not None else False)
    valid_outlier_mask = df['valid'] == -1
    match = bool((z_outlier_mask.values == valid_outlier_mask.values).all())

    file_start, file_end = df['Time'].iloc[0], df['Time'].iloc[-1]
    manual = resolve_manual_window(args, file_start, file_end)
    if manual is not None:
        win_start, win_end = manual
        window_mode = 'manual'
    elif valid_outlier_mask.any() and args.auto_window:
        event_time = select_representative_outlier(
            df.loc[valid_outlier_mask, 'Time'], df.loc[valid_outlier_mask, '_z'],
            file_start, file_end)
        pad = pd.Timedelta(minutes=compute_adaptive_pad_minutes(df))
        win_start = max(file_start, event_time - pad)
        win_end = min(file_end, event_time + pad)
        window_mode = 'auto'
    else:
        win_start, win_end = file_start, file_end
        window_mode = 'full-file (no outliers / auto-window disabled)'

    win_mask = (df['Time'] >= win_start) & (df['Time'] <= win_end)
    df_win = df.loc[win_mask].reset_index(drop=True)
    out = df_win['valid'] == -1
    inl = ~out
    n_shown_outliers = int(out.sum())

    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.6), constrained_layout=True, sharex=True)

    # --- Panel (a): signal ---
    line = downsample_for_line(df_win.loc[inl], value_col, target_points=400)
    ax1.plot(line['Time'], line[value_col], color=MUTED_SIGNAL_COLOR, lw=1.0, alpha=0.9, label='Valid')
    if n_shown_outliers:
        ax1.scatter(df_win.loc[out, 'Time'], df_win.loc[out, value_col],
                    color=VALID_COLORS[-1], marker='x', s=42, linewidths=1.5, zorder=5,
                    label='Global outlier')

    normal_vals = line[value_col]
    lo, hi = float(normal_vals.min()), float(normal_vals.max())
    span = hi - lo
    margin = span * 0.15 if span > 0 else max(abs(hi), 1.0) * 0.05
    normal_ylim = (lo - margin, hi + margin)

    needs_inset_a = False
    if n_shown_outliers:
        out_vals = df_win.loc[out, value_col]
        needs_inset_a = bool(((out_vals < normal_ylim[0]) | (out_vals > normal_ylim[1])).any())

    if needs_inset_a:
        axins = ax1.inset_axes((0.58, 0.60, 0.39, 0.36))
        axins.plot(line['Time'], line[value_col], color=MUTED_SIGNAL_COLOR, lw=0.8)
        axins.set_xlim(line['Time'].min(), line['Time'].max())
        axins.set_ylim(*normal_ylim)
        axins.tick_params(labelsize=6, labelbottom=False)
        axins.set_title('normal range', fontsize=6.5, pad=1.5)
        sci_y_axis(axins)
        for spine in axins.spines.values():
            spine.set_linewidth(0.6)
    else:
        ax1.set_ylim(*normal_ylim)

    ax1.set_ylabel(axis_label(value_col))
    ax1.grid(True, alpha=0.2, linewidth=0.4)
    sci_y_axis(ax1)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.85, fontsize=7.5,
               handlelength=1.4, borderaxespad=0.4)
    panel_label(ax1, '(a)')

    # --- Panel (b): robust Z-score ---
    z_line = downsample_for_line(
        df_win.loc[inl, ['Time']].assign(_zval=df_win.loc[inl, '_z']), '_zval', target_points=400)
    ax2.plot(z_line['Time'], z_line['_zval'], color='#555555', lw=1.0, alpha=0.9)
    if n_shown_outliers:
        ax2.scatter(df_win.loc[out, 'Time'], df_win.loc[out, '_z'],
                    color=VALID_COLORS[-1], marker='x', s=42, linewidths=1.5, zorder=5)

    # y in data coords, x in axes-fraction — keeps the label fully inside the
    # panel regardless of the window width (a plain data-coord placement at
    # win_end got clipped by the axes edge).
    thr_color = VALID_COLORS[-1]
    yaxis_t = ax2.get_yaxis_transform()
    ax2.axhline(z_low, color=thr_color, linestyle='--', linewidth=1.1)
    ax2.text(0.985, z_low, f'{z_low:g} ', transform=yaxis_t, color=thr_color,
              fontsize=7, va='bottom', ha='right')
    if z_high is not None:
        ax2.axhline(z_high, color=thr_color, linestyle='--', linewidth=1.1)
        ax2.text(0.985, z_high, f'{z_high:g} ', transform=yaxis_t, color=thr_color,
                  fontsize=7, va='bottom', ha='right')

    normal_z = z_line['_zval']
    zlo = min(float(normal_z.min()), z_low) - 1.0
    zhi_ref = z_high if z_high is not None else float(normal_z.max())
    zhi = max(float(normal_z.max()), zhi_ref) + 1.0

    needs_inset_b = False
    if n_shown_outliers:
        oz = df_win.loc[out, '_z']
        needs_inset_b = bool(((oz < zlo) | (oz > zhi)).any())

    if needs_inset_b:
        axins2 = ax2.inset_axes((0.58, 0.08, 0.39, 0.40))
        axins2.plot(z_line['Time'], z_line['_zval'], color='#555555', lw=0.8)
        axins2.axhline(z_low, color=thr_color, linestyle='--', linewidth=1.0)
        if z_high is not None:
            axins2.axhline(z_high, color=thr_color, linestyle='--', linewidth=1.0)
        axins2.set_xlim(z_line['Time'].min(), z_line['Time'].max())
        axins2.set_ylim(zlo, zhi)
        axins2.tick_params(labelsize=6, labelbottom=False)
        axins2.set_title('threshold region', fontsize=6.5, pad=1.5)
        for spine in axins2.spines.values():
            spine.set_linewidth(0.6)
    else:
        ax2.set_ylim(zlo, zhi)

    threshold_handle = Line2D([0], [0], color=thr_color, linestyle='--', linewidth=1.1, label='Threshold')
    ax2.set_ylabel('Robust Z-score')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.2, linewidth=0.4)
    ax2.legend(handles=[threshold_handle], loc='lower left', frameon=True,
               framealpha=0.85, fontsize=7.5, handlelength=1.4, borderaxespad=0.4)
    format_time_axis(ax2)
    panel_label(ax2, '(b)')

    save_fig(fig, 'global_outlier_detection', args.figures_dir)
    plt.close(fig)

    print_repro('Figure 2 reproducibility info', [
        ('series', series_dir.name),
        ('filename', target.stem),
        ('displayed feature', value_col),
        ('time range (full file)', f'{file_start} -> {file_end}'),
        ('time range (displayed window)', f'{win_start} -> {win_end}'),
        ('window mode', window_mode),
        ('global outliers shown in window', n_shown_outliers),
        ('global outliers (valid==-1, full file)', int((df['valid'] == -1).sum())),
        ('local outliers (valid==0, full file)', int((df['valid'] == 0).sum())),
        ('valid samples (valid==1, full file)', int((df['valid'] == 1).sum())),
        ('z_low / z_high (clean.json global_outlier)', f'{z_low} / {z_high}'),
        ('recomputed-Z vs stored valid==-1 exact match', match),
    ])


# =============================================================================
# FIGURE 3 — LOCAL OUTLIERS + CLEANED SIGNAL
# =============================================================================

def plot_local_cleaning_figure(args, clean_cfg: dict) -> None:
    print('\n=== Figure 3: Local outlier detection and cleaned signal ===')
    series_dir, target = select_outlier_example(
        args.data_root, args.house, args.series, args.file, kind='local')

    df = load_full(target)
    value_col = args.value_col
    time_col = 'Time'
    global_cfg = clean_cfg['global_outlier']
    local_cfg = clean_cfg['local_outlier']
    block_size = local_cfg['block_size']

    # Reproduce label_outliers()'s Stage 1 (global) + Stage 2 (local) exactly,
    # over the FULL file, using the same functions the pipeline calls. Window
    # selection below only decides what gets *displayed* — it never touches
    # this classification.
    z = robust_zscore(df[value_col])
    global_mask = (z < global_cfg['z_low']) | (
        (z > global_cfg['z_high']) if global_cfg['z_high'] is not None else False)
    df_in = df.loc[~global_mask].sort_values(time_col).reset_index(drop=True).copy()
    df_in['_block'] = df_in.index // block_size

    blocks = compute_local_outlier_blocks(
        df_in, value_col=value_col, time_col=time_col, block_size=block_size,
        min_run_blocks=local_cfg['min_run_blocks'],
        step_up_quantile=local_cfg['step_up_quantile'],
        smooth_window_blocks=local_cfg['smooth_window_blocks'],
        k_block=local_cfg['k_block'], center_smooth=local_cfg['center_smooth'],
    )
    block_bounds = df_in.groupby('_block')[time_col].agg(t0='min', t1='max')
    blocks = blocks.merge(block_bounds, left_on='_block', right_index=True).reset_index(drop=True)

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    file_start, file_end = df_sorted[time_col].iloc[0], df_sorted[time_col].iloc[-1]

    manual = resolve_manual_window(args, file_start, file_end)
    invalid_lo_hi = None  # (block-index start, end) of the highlighted invalid run, if any
    if manual is not None:
        win_start, win_end = manual
        blocks_win = blocks[(blocks['t1'] >= win_start) & (blocks['t0'] <= win_end)].reset_index(drop=True)
        window_mode = 'manual'
    else:
        invalid_mask = ~blocks['keep_block'].to_numpy()
        max_invalid_len = max(1, args.max_blocks - 6)  # leave room for context on both sides
        run = _find_isolated_run(invalid_mask, min_len=1, max_len=max_invalid_len)
        if run is None:
            win_start, win_end = file_start, file_end
            blocks_win = blocks
        else:
            b_start, b_end = run
            invalid_lo_hi = (b_start, b_end)
            context = max(2, (args.max_blocks - (b_end - b_start + 1)) // 2)
            lo, hi = max(0, b_start - context), min(len(blocks) - 1, b_end + context)
            while (hi - lo + 1) > args.max_blocks and (hi > b_end or lo < b_start):
                if hi > b_end:
                    hi -= 1
                elif lo < b_start:
                    lo += 1
            win_start, win_end = blocks.iloc[lo]['t0'], blocks.iloc[hi]['t1']
            blocks_win = blocks.iloc[lo:hi + 1].reset_index(drop=True)
        window_mode = 'auto'

    n_blocks_shown = len(blocks_win)
    good = blocks_win['keep_block']
    n_invalid_blocks_shown = int((~good).sum())

    invalid_span = None
    if invalid_lo_hi is not None:
        b_start, b_end = invalid_lo_hi
        invalid_span = (blocks.iloc[b_start]['t0'], blocks.iloc[b_end]['t1'])
    elif n_invalid_blocks_shown > 0:
        inv = blocks_win.loc[~good]
        invalid_span = (inv['t0'].min(), inv['t1'].max())

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.8, 8.8), constrained_layout=True, sharex=True)

    # --- Panel (a): raw signal ---
    df_win_raw = crop(df_sorted, win_start, win_end)
    ax1.plot(df_win_raw[time_col], df_win_raw[value_col], color='#777777',
              lw=0.9, alpha=0.85, rasterized=True)
    for k, (_, row) in enumerate(blocks_win.iterrows()):
        if k % 2 == 0:
            ax1.axvspan(row['t0'], row['t1'], color='#888888', alpha=0.06, lw=0)
    if invalid_span is not None:
        ax1.axvspan(invalid_span[0], invalid_span[1], color=VALID_COLORS[0], alpha=0.18, lw=0, zorder=2)
        mid = invalid_span[0] + (invalid_span[1] - invalid_span[0]) / 2
        ax1.annotate('invalid block(s)', xy=(mid, 1.0), xycoords=('data', 'axes fraction'),
                     xytext=(0, 2), textcoords='offset points', ha='center', va='bottom',
                     fontsize=7, color=VALID_COLORS[0])

    # isolated point-level outliers inside otherwise-good (kept) blocks
    kept_block_ids = set(blocks_win.loc[good, '_block'])
    df_win_in = df_in[(df_in[time_col] >= win_start) & (df_in[time_col] <= win_end)]
    isolated_pts = df_win_in[df_win_in['_block'].isin(kept_block_ids) & (df_win_in['valid'] == 0)]
    if len(isolated_pts):
        ax1.scatter(isolated_pts[time_col], isolated_pts[value_col], color=VALID_COLORS[0],
                    s=10, zorder=4, label='Local outlier')
        ax1.legend(loc='upper right', frameon=True, framealpha=0.85, fontsize=7.5,
                   handlelength=1.4, borderaxespad=0.4)
    ax1.set_ylabel(axis_label(value_col))
    ax1.grid(True, alpha=0.2, linewidth=0.4)
    sci_y_axis(ax1)
    panel_label(ax1, '(a)')

    # --- Panel (b): block means + rolling-median reference + threshold band ---
    ax2.plot(blocks_win['block_time'], blocks_win['block_mean'], color='#999999',
              lw=0.7, alpha=0.7, zorder=1)
    ax2.fill_between(blocks_win['block_time'],
                      blocks_win['smooth'] - local_cfg['k_block'] * blocks_win['block_scale'],
                      blocks_win['smooth'] + local_cfg['k_block'] * blocks_win['block_scale'],
                      color=MUTED_SIGNAL_COLOR, alpha=0.15, zorder=1, label='Threshold')
    ax2.plot(blocks_win['block_time'], blocks_win['smooth'], color=MUTED_SIGNAL_COLOR,
              lw=1.6, zorder=3, label='Rolling median')
    ax2.scatter(blocks_win.loc[good, 'block_time'], blocks_win.loc[good, 'block_mean'],
                color=VALID_COLORS[1], s=12, zorder=4, label='Valid')

    invalid_blocks = blocks_win.loc[~good]
    if 0 < n_invalid_blocks_shown <= 5:
        ax2.scatter(invalid_blocks['block_time'], invalid_blocks['block_mean'],
                    color=VALID_COLORS[0], marker='x', s=55, linewidths=1.6, zorder=5,
                    label='Local outlier')
    elif n_invalid_blocks_shown > 5:
        if invalid_span is not None:
            ax2.axvspan(invalid_span[0], invalid_span[1], color=VALID_COLORS[0], alpha=0.12, lw=0, zorder=2)
        pick = [invalid_blocks.index[0], invalid_blocks.index[len(invalid_blocks) // 2], invalid_blocks.index[-1]]
        ax2.scatter(blocks_win.loc[pick, 'block_time'], blocks_win.loc[pick, 'block_mean'],
                    color=VALID_COLORS[0], marker='x', s=55, linewidths=1.6, zorder=5,
                    label='Local outlier')

    ax2.set_ylabel(f'{axis_label(value_col)}\n(block mean)')
    ax2.grid(True, alpha=0.2, linewidth=0.4)
    sci_y_axis(ax2)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ['Valid', 'Local outlier', 'Rolling median', 'Threshold']
    ax2.legend([by_label[k] for k in order if k in by_label], [k for k in order if k in by_label],
               loc='upper right', frameon=True, framealpha=0.85, fontsize=7.5,
               handlelength=1.4, borderaxespad=0.4, ncol=2, columnspacing=0.8)
    panel_label(ax2, '(b)')

    # --- Panel (c): cleaning result — continuous line, NaN-gapped at removed points ---
    df_win = crop(df_sorted, win_start, win_end)
    cleaned = df_win[value_col].where(df_win['valid'] == 1)
    ax3.plot(df_win[time_col], df_win[value_col], color='#cccccc', lw=0.8, alpha=0.7, zorder=1)
    ax3.plot(df_win[time_col], cleaned, color=VALID_COLORS[1], lw=1.1, zorder=3, label='Valid')
    if invalid_span is not None:
        ax3.axvspan(invalid_span[0], invalid_span[1], color=VALID_COLORS[0], alpha=0.12, lw=0, zorder=2)
    ax3.set_ylabel(axis_label(value_col))
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.2, linewidth=0.4)
    sci_y_axis(ax3)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.85, fontsize=7.5,
               handlelength=1.4, borderaxespad=0.4)
    format_time_axis(ax3)
    panel_label(ax3, '(c)')

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(win_start, win_end)

    save_fig(fig, 'local_outlier_cleaning', args.figures_dir)
    plt.close(fig)

    # Sanity check: recomputed local-outlier classification (full file) should
    # match the stored 'valid' column exactly (same function, same config).
    df_in_check = df_in.merge(blocks[['_block', 'keep_block', 'smooth']], on='_block', how='left')
    df_in_check['resid_point'] = df_in_check[value_col] - df_in_check['smooth']
    scale_point = robust_mad_scale(df_in_check.loc[df_in_check['keep_block'], 'resid_point'])
    good_point = df_in_check['resid_point'].abs() <= (local_cfg['k_point'] * scale_point)
    recomputed_valid_local = np.where(
        (~df_in_check['keep_block']) | (~good_point), 0, 1)
    stored_valid_local = df_in_check['valid'].to_numpy()
    match = bool((recomputed_valid_local == stored_valid_local).all())

    print_repro('Figure 3 reproducibility info', [
        ('series', series_dir.name),
        ('filename', target.stem),
        ('displayed feature', value_col),
        ('time range (full file)', f'{file_start} -> {file_end}'),
        ('time range (displayed window)', f'{win_start} -> {win_end}'),
        ('window mode', window_mode),
        ('blocks shown', n_blocks_shown),
        ('invalid blocks shown', n_invalid_blocks_shown),
        ('isolated local-outlier points shown (inside kept blocks)', int(len(isolated_pts))),
        ('global outliers (valid==-1, full file)', int((df['valid'] == -1).sum())),
        ('local outliers (valid==0, full file)', int((df['valid'] == 0).sum())),
        ('valid samples (valid==1, full file)', int((df['valid'] == 1).sum())),
        ('block_size', block_size),
        ('min_run_blocks', local_cfg['min_run_blocks']),
        ('step_up_quantile', local_cfg['step_up_quantile']),
        ('smooth_window_blocks', local_cfg['smooth_window_blocks']),
        ('k_block', local_cfg['k_block']),
        ('k_point', local_cfg['k_point']),
        ('center_smooth', local_cfg['center_smooth']),
        ('recomputed vs stored valid (inlier subset) exact match', match),
    ])


# =============================================================================
# MAIN
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plot', choices=['labelling', 'global-outliers', 'local-cleaning', 'all'],
                    default='all', help='Which figure(s) to generate.')
    p.add_argument('--data-root', type=str, default='data/processed',
                    help='Root directory of processed series (default: data/processed).')
    p.add_argument('--config-dir', type=str, default='configs/preprocessing',
                    help='Directory with clean.json.')
    p.add_argument('--figures-dir', type=str, default='figures',
                    help='Output directory for the figures.')
    p.add_argument('--value-col', type=str, default=None,
                    help='Feature column to plot (default: clean.json value_col, '
                         '"t_up_0 in ps").')
    p.add_argument('--house', choices=['A', 'B', 'any'], default='A',
                    help='House preference for automatic selection (default: A). '
                         'Falls back to the other house if none qualifies.')
    p.add_argument('--series', type=str, default=None,
                    help='Manual series override (directory name under --data-root).')
    p.add_argument('--file', type=str, default=None,
                    help='Manual file override (stem or filename, requires --series).')
    p.add_argument('--focus-hours', type=float, default=None,
                    help='Labelling figure only: padding (hours) around the leak window '
                         f'for cropping the displayed window (default {DEFAULT_FOCUS_HOURS_LABELLING}). '
                         'Global/local-outlier figures use --auto-window / --window-start / '
                         '--window-end / --max-blocks instead.')
    p.add_argument('--window-start', type=str, default=None,
                    help='Global/local-outlier figures: manual display-window start '
                         '(ISO8601 timestamp). Must be given together with --window-end; '
                         'overrides automatic window selection.')
    p.add_argument('--window-end', type=str, default=None,
                    help='Global/local-outlier figures: manual display-window end '
                         '(ISO8601 timestamp). Must be given together with --window-start.')
    p.add_argument('--auto-window', action=argparse.BooleanOptionalAction, default=True,
                    help='Global/local-outlier figures: automatically select a short, '
                         'representative display window (default: on). Use --no-auto-window '
                         'with --window-start/--window-end for full manual control, or alone '
                         'to show the complete recording.')
    p.add_argument('--max-blocks', type=int, default=DEFAULT_MAX_BLOCKS,
                    help='Local-outlier figure: maximum number of 64-sample blocks to '
                         f'include in the auto-selected display window (default {DEFAULT_MAX_BLOCKS}, '
                         'roughly 10-30 recommended).')
    return p


def main():
    args = build_arg_parser().parse_args()
    args.data_root = Path(args.data_root)
    args.config_dir = Path(args.config_dir)
    args.figures_dir = Path(args.figures_dir)

    clean_cfg = load_clean_config(args.config_dir)
    if args.value_col is None:
        args.value_col = clean_cfg['value_col']

    if not args.data_root.exists():
        raise RuntimeError(f'--data-root {args.data_root} does not exist.')

    targets = ['labelling', 'global-outliers', 'local-cleaning'] if args.plot == 'all' else [args.plot]
    args.focus_hours = args.focus_hours if args.focus_hours is not None else DEFAULT_FOCUS_HOURS_LABELLING
    failures = []

    for target in targets:
        try:
            if target == 'labelling':
                plot_labelling_figure(args)
            elif target == 'global-outliers':
                plot_global_outliers_figure(args, clean_cfg)
            elif target == 'local-cleaning':
                plot_local_cleaning_figure(args, clean_cfg)
        except RuntimeError as e:
            print(f'  [FAILED] {target}: {e}')
            failures.append(target)
            if args.plot != 'all':
                raise

    if failures:
        print(f'\n{len(failures)}/{len(targets)} figure(s) failed: {failures}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
