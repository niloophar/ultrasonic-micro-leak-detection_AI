#!/usr/bin/env python
"""
Thesis figure — single-panel preprocessing overview: one full recording with
valid measurements, local outliers, global outliers, and the leakage window
all shown together.

Reuses the actual pipeline output and logic — no recomputation of the
outlier classification:
  - ``valid`` column (1=valid, 0=local outlier, -1=global outlier) comes
    straight from ``data/processed/<series>/<file>.csv``, written by
    ``src/preprocessing/clean.py::label_outliers``.
  - Leakage start/end come from ``get_leakage_times`` in
    ``src/common/plotting.py``, which derives them from the ``flow_type``
    column written by ``src/preprocessing/label.py`` — the same helper the
    pipeline's own diagnostic plots call.
  - ``t_up_0 in ps`` is the default displayed feature, matching
    ``clean.json``'s ``value_col`` (the column the pipeline itself uses for
    outlier detection).

This script never writes to data/processed and never invents thresholds or
data.

Run (needs the full data/processed/<series>/*.csv tree — see CLAUDE.md's
"Development workflow"):

    python scripts/plot_preprocessing_outlier_overview.py

Outputs:
    figures/preprocessing_outlier_overview.{pdf,png}
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from src.common.plotting import get_leakage_times

# =============================================================================
# STYLE
# =============================================================================

VALID_COLOR = '#1f5fa8'    # blue
LOCAL_COLOR = '#e08214'    # orange
GLOBAL_COLOR = '#d62728'   # red
LEAKAGE_COLOR = '#2ca02c'  # green

LW_VALID = 0.6
S_LOCAL = 5
S_GLOBAL = 32

DEFAULT_FEATURE = 't_up_0 in ps'


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
    """'t_up_0 in ps' -> 't_up_0 (ps)'; passes through columns without ' in '."""
    if ' in ' in col:
        name, unit = col.split(' in ', 1)
        return f'{name} ({unit})'
    return col


def format_time_axis(ax) -> None:
    """Adaptive, non-repetitive tick resolution (ConciseDateFormatter) so a
    full-recording x-axis doesn't repeat the date on every tick."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def sci_y_axis(ax) -> None:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)


def display_filename(stem: str) -> str:
    """Drop underscores from a raw filename stem for the figure title, e.g.
    'ha_2026_04_18 1800_1019_no_leakage' -> 'ha 2026 04 18 1800 1019 no leakage'."""
    return ' '.join(stem.replace('_', ' ').split())


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


def house_passes(house: str) -> List[str]:
    if house == 'A':
        return ['A', 'B']
    if house == 'B':
        return ['B', 'A']
    return ['any']


def scan_counts(path: Path) -> Optional[Tuple[int, int, int, bool]]:
    """Return (n_valid, n_local, n_global, has_leakage) for one CSV, or None
    if the required columns aren't present."""
    try:
        df = pd.read_csv(path, usecols=['valid', 'flow_type'])
    except (ValueError, pd.errors.EmptyDataError):
        return None
    n_valid = int((df['valid'] == 1).sum())
    n_local = int((df['valid'] == 0).sum())
    n_global = int((df['valid'] == -1).sum())
    has_leak = bool((df['flow_type'] == 'flow').any())
    return n_valid, n_local, n_global, has_leak


def select_overview_file(
    data_root: Path, house: str, manual_series: Optional[str], manual_file: Optional[str]
) -> Tuple[Path, Path]:
    """Pick a recording containing valid, local-outlier, global-outlier, and
    leakage points — the four ingredients this figure needs to be meaningful.
    House A is preferred (see --house); falls back to House B only if no
    House A recording qualifies."""
    if manual_series and manual_file:
        series_dir = data_root / manual_series
        target = series_dir / manual_file
        if not target.suffix:
            target = series_dir / f'{manual_file}.csv'
        if not target.exists():
            raise RuntimeError(f'Manual file not found: {target}')
        return series_dir, target

    for house_pass in house_passes(house):
        candidates = ordered_series_dirs(data_root, house_pass if house_pass != 'any' else house)
        best = None  # (score, series_dir, file)
        for series_dir in candidates:
            if house_pass in ('A', 'B'):
                if house_pass == 'A' and is_house_b(series_dir.name):
                    continue
                if house_pass == 'B' and not is_house_b(series_dir.name):
                    continue
            for f in sorted(series_dir.glob('*.csv')):
                counts = scan_counts(f)
                if counts is None:
                    continue
                n_valid, n_local, n_global, has_leak = counts
                if not (n_valid > 0 and n_local > 0 and n_global > 0 and has_leak):
                    continue
                # Reward a visible-but-not-overwhelming number of each
                # outlier type so the three point classes stay distinguishable
                # rather than one drowning out the others.
                score = min(n_local, 200) + min(n_global, 200)
                if best is None or score > best[0]:
                    best = (score, series_dir, f)
        if best is not None:
            return best[1], best[2]

    raise RuntimeError(
        'No recording with valid + local-outlier + global-outlier points AND '
        f'a leakage window found under {data_root}. Sync more data/processed '
        '<series> directories from the GPU server (see CLAUDE.md '
        '"Development workflow"), or select a file manually with --series/--file.'
    )


def build_legend(ax, n_valid: int, n_local: int, n_global: int,
                  leak_start: Optional[pd.Timestamp], leak_end: Optional[pd.Timestamp]) -> None:
    handles = [
        Line2D([0], [0], color=VALID_COLOR, lw=1.6,
               label=f'Valid (N={n_valid:,})'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=LOCAL_COLOR,
               markeredgecolor='none', markersize=5,
               label=f'Local outliers (N={n_local:,})'),
        Line2D([0], [0], marker='x', color=GLOBAL_COLOR, linestyle='none',
               markeredgewidth=1.4, markersize=7,
               label=f'Global outliers (N={n_global:,})'),
    ]
    if leak_start is not None and leak_end is not None:
        handles.append(Line2D([0], [0], color=LEAKAGE_COLOR, linestyle='--', lw=1.2,
                               label=f'Leakage start ({leak_start.strftime("%H:%M")})'))
        handles.append(Line2D([0], [0], color=LEAKAGE_COLOR, linestyle='--', lw=1.2,
                               label=f'Leakage end ({leak_end.strftime("%H:%M")})'))
    ax.legend(handles=handles, loc='upper right', frameon=True, framealpha=0.88,
               fontsize=7.5, handlelength=1.6, borderaxespad=0.6)


def plot_overview(args) -> None:
    print('\n=== Preprocessing outlier overview ===')
    series_dir, target = select_overview_file(
        args.data_root, args.house, args.series, args.file)

    df = pd.read_csv(target)
    if args.feature not in df.columns:
        raise RuntimeError(
            f'Feature column {args.feature!r} not found in {target}. '
            f'Available columns: {list(df.columns)}')
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    value_col = args.feature
    leak_start, leak_end = get_leakage_times(df, {})

    n_total = len(df)
    n_valid = int((df['valid'] == 1).sum())
    n_local = int((df['valid'] == 0).sum())
    n_global = int((df['valid'] == -1).sum())

    apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)

    # Valid measurements — thin line, gapped at removed points so the line
    # never bridges across an outlier run. Rasterized: a full-recording trace
    # can be hundreds of thousands of points; PDF text/axes stay vector.
    valid_trace = df[value_col].where(df['valid'] == 1)
    ax.plot(df['Time'], valid_trace, color=VALID_COLOR, lw=LW_VALID,
            alpha=0.9, rasterized=True, zorder=2)

    # Local outliers — small, semi-transparent, dense-cloud rasterized.
    local_df = df[df['valid'] == 0]
    if len(local_df):
        ax.scatter(local_df['Time'], local_df[value_col], color=LOCAL_COLOR,
                   s=S_LOCAL, alpha=0.55, linewidths=0, zorder=3, rasterized=True)

    # Global outliers — sparse, kept vector, higher zorder, larger x markers.
    global_df = df[df['valid'] == -1]
    if len(global_df):
        ax.scatter(global_df['Time'], global_df[value_col], color=GLOBAL_COLOR,
                   marker='x', s=S_GLOBAL, linewidths=1.4, zorder=5)

    if leak_start is not None:
        ax.axvline(leak_start, color=LEAKAGE_COLOR, linestyle='--', lw=1.2, zorder=4)
    if leak_end is not None:
        ax.axvline(leak_end, color=LEAKAGE_COLOR, linestyle='--', lw=1.2, zorder=4)

    title = f'{series_dir.name} – {display_filename(target.stem)} – All measurements'
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(axis_label(value_col))
    ax.grid(True, color='#cccccc', alpha=0.35, linewidth=0.5)
    sci_y_axis(ax)
    format_time_axis(ax)
    build_legend(ax, n_valid, n_local, n_global, leak_start, leak_end)

    save_fig(fig, 'preprocessing_outlier_overview', args.output_dir)
    plt.close(fig)

    print(f'  series:              {series_dir.name}')
    print(f'  source file:         {target.name}')
    print(f'  displayed feature:   {value_col}')
    print(f'  total measurements:  {n_total:,}')
    print(f'  valid measurements:  {n_valid:,}')
    print(f'  local outliers:      {n_local:,}')
    print(f'  global outliers:     {n_global:,}')
    print(f'  leakage start:       {leak_start.strftime("%H:%M") if leak_start is not None else "n/a"}')
    print(f'  leakage end:         {leak_end.strftime("%H:%M") if leak_end is not None else "n/a"}')


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--series', type=str, default=None,
                    help='Manual series override (directory name under --data-root). '
                         'Must be given together with --file.')
    p.add_argument('--file', type=str, default=None,
                    help='Manual file override (stem or filename, requires --series).')
    p.add_argument('--feature', type=str, default=None,
                    help=f'Feature column to plot (default: clean.json value_col, '
                         f'"{DEFAULT_FEATURE}").')
    p.add_argument('--auto-select', action=argparse.BooleanOptionalAction, default=True,
                    help='Automatically select a representative recording when --series/'
                         '--file are not both given (default: on).')
    p.add_argument('--data-root', type=str, default='data/processed',
                    help='Root directory of processed series (default: data/processed).')
    p.add_argument('--config-dir', type=str, default='configs/preprocessing',
                    help='Directory with clean.json (used only to resolve the default feature).')
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
    if not args.series and not args.auto_select:
        raise RuntimeError('--no-auto-select requires --series and --file.')

    if args.feature is None:
        with open(args.config_dir / 'clean.json') as f:
            clean_cfg = json.load(f)
        args.feature = clean_cfg.get('value_col', DEFAULT_FEATURE)

    if not args.data_root.exists():
        raise RuntimeError(f'--data-root {args.data_root} does not exist.')

    plot_overview(args)


if __name__ == '__main__':
    main()
