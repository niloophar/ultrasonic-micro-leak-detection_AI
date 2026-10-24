#!/usr/bin/env python
"""
Thesis figure — NoFlow baseline drift + temperature, House A vs House B (Chapter 3).

Polished, per-series version of the exploratory "drift_hb"-style plots in
`src/common/plotting.py` (plot_drift_overview_files_subplots). Keeps the
reference's information density -- one colour per series, dense per-block
background scatter, per-file mean markers, per-house trend lines, a
multi-column legend above the plot -- but:
  - restricts everything to flow_type == 'noflow' & valid == 1
    (& equilibrium == 1 if present); the reference used all valid rows
    (flow included), which is why its bands are wider than the true
    NoFlow baseline.
  - swaps the degree-3 polynomial per-house trend for LOWESS.
  - uses a 24-colour palette (tab20 + 4 from tab20b) instead of tab10,
    which repeated every 10 series in the reference.
  - flags 30.03.2026 as excluded (estimated labels, see CLAUDE.md) and
    drops it from the House A trend fit.
  - no in-figure titles (LaTeX caption explains the figure).

Dataset-characterisation only -- no model performance, no GMM/AUC content.

Outputs:
  figures/dataset_drift_overview.pdf
  figures/dataset_drift_overview.png

Run (needs the full data/processed/<series>/*.csv tree -- typically on the
GPU server, see commands.txt):
    python scripts/plot_thesis_dataset_drift_overview.py
"""
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

VALUE_COL = 't_up_0 in ps'
TEMP_COL = 'Temp Ambient in °C'
TIME_COL = 'Time'
BLOCK_SIZE = 512

EXCLUDED_SERIES = {'30.03.2026'}

SERIES = [
    # House A
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026', '7.03.2026',
    '15.03.2026', '30.03.2026', '5.04.2026', '11.04.2026', '18.04.2026',
    'ha-25.04.2026', 'ha-2.05.2026', 'ha-9.05.2026', 'ha-17.05.2026',
    'ha-24.05.2026', 'ha-2.06.2026', 'ha-6.06.2026',
    # House B
    'hb-26.04.2026', 'hb-4.05.2026', 'hb-10.05.2026', 'hb-17.05.2026',
    'hb-25.05.2026', 'hb-31.05.2026', 'hb-7.06.2026',
]

MARKER_A = 'D'
MARKER_B = '^'
TREND_COLOR_A = '#e8871e'   # orange
TREND_COLOR_B = '#1f77b4'   # blue

WANTED_COLS = {TIME_COL, VALUE_COL, TEMP_COL, 'flow_type', 'valid', 'equilibrium'}


def house_of(series_name: str) -> str:
    return 'B' if series_name.startswith('hb-') else 'A'


def series_palette(n: int) -> list:
    base = list(plt.cm.tab20.colors)
    extra = list(plt.cm.tab20b.colors)
    colors = (base + extra)[:n]
    return colors


def compute_series_data(series_dir: Path, series_name: str) -> dict:
    """Block-averaged background points + per-file medians, NoFlow-equilibrium only."""
    block_times, block_values, block_temps = [], [], []
    file_points = []

    for csv_file in sorted(series_dir.glob('*.csv')):
        try:
            df = pd.read_csv(csv_file, usecols=lambda c: c in WANTED_COLS)
        except Exception as e:
            print(f'    [skip file] {csv_file.name}: {e}')
            continue
        if not {'flow_type', 'valid', TIME_COL, VALUE_COL}.issubset(df.columns):
            continue

        mask = (df['flow_type'] == 'noflow') & (df['valid'] == 1)
        if 'equilibrium' in df.columns:
            mask &= (df['equilibrium'] == 1)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue

        sub[TIME_COL] = pd.to_datetime(sub[TIME_COL])
        sub = sub.sort_values(TIME_COL).reset_index(drop=True)
        has_temp = TEMP_COL in sub.columns

        sub['_block'] = sub.index // BLOCK_SIZE
        agg_kwargs = {TIME_COL: (TIME_COL, 'mean'), VALUE_COL: (VALUE_COL, 'mean')}
        if has_temp:
            agg_kwargs[TEMP_COL] = (TEMP_COL, 'mean')
        block_avg = sub.groupby('_block', as_index=False).agg(**agg_kwargs)

        block_times.extend(block_avg[TIME_COL].tolist())
        block_values.extend(block_avg[VALUE_COL].tolist())
        if has_temp:
            block_temps.extend(block_avg[TEMP_COL].tolist())

        file_points.append({
            'time': sub[TIME_COL].median(),
            'value': float(sub[VALUE_COL].median()),
            'temp': float(sub[TEMP_COL].median()) if has_temp else np.nan,
        })

    return {
        'name': series_name,
        'house': house_of(series_name),
        'excluded': series_name in EXCLUDED_SERIES,
        'block_times': block_times,
        'block_values': block_values,
        'block_temps': block_temps,
        'files': file_points,
    }


def collect_series(processed_root: Path) -> list:
    series_data = []
    for series_name in SERIES:
        series_dir = processed_root / series_name
        if not series_dir.exists():
            print(f'  [skip] {series_name} — not found under {processed_root}')
            continue
        sd = compute_series_data(series_dir, series_name)
        print(f'  {series_name}: {len(sd["files"])} files, {len(sd["block_values"])} blocks')
        if sd['files']:
            series_data.append(sd)
    return series_data


def lowess_smooth(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray,
                   frac: float = 0.15, min_points: int = 5) -> np.ndarray:
    """Tricube-weighted local-linear smoother (no statsmodels dependency),
    bandwidth = max(fixed fraction of the time span, distance to the
    min_points-th nearest sample).

    A pure nearest-neighbour-count bandwidth (textbook LOWESS) is unstable
    across multi-week gaps -- House B has several -- because the k-th
    nearest neighbour can suddenly jump from a tight local cluster to a
    point across the gap, producing a spurious kink. A pure fixed bandwidth
    collapses to near-zero weight in gaps wider than the bandwidth, which
    produces an ill-conditioned fit (or a flat/step artifact if you fall
    back to nearest-value). Taking the max of both: dense regions use the
    fine fixed-fraction bandwidth, and it widens continuously (not in a
    step) inside gaps just enough to keep at least min_points samples
    in view, so the fit stays well-determined and varies smoothly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = x.max() - x.min()
    h_fixed = max(frac * span, 1e-8)
    n = len(x)
    k = min(min_points, n)
    y_eval = np.empty(len(x_eval))
    ones = np.ones(n)
    A = np.column_stack([ones, x])
    for j, xe in enumerate(x_eval):
        dist = np.abs(x - xe)
        h_adaptive = np.sort(dist)[k - 1]
        h = max(h_fixed, h_adaptive, 1e-8)
        w = np.clip(dist / h, 0, 1)
        w = (1 - w ** 3) ** 3
        Aw = A * w[:, None]
        ATA = Aw.T @ A
        ATy = Aw.T @ y
        try:
            beta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(ATA, ATy, rcond=None)[0]
        y_eval[j] = beta[0] + beta[1] * xe
    return y_eval


def house_trend(series_data: list, house: str, value_key: str):
    """LOWESS trend over all per-file points of one house, excluding flagged series."""
    times, values = [], []
    for sd in series_data:
        if sd['house'] != house or sd['excluded']:
            continue
        for fp in sd['files']:
            v = fp[value_key]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            times.append(fp['time'])
            values.append(v)
    if len(times) < 4:
        return None, None
    x_num = mdates.date2num(pd.to_datetime(times).to_pydatetime())
    y = np.asarray(values, dtype=float)
    order = np.argsort(x_num)
    x_num, y = x_num[order], y[order]
    x_eval = np.linspace(x_num.min(), x_num.max(), 200)
    y_eval = lowess_smooth(x_num, y, x_eval, frac=0.15)
    return x_eval, y_eval  # x_eval stays numeric -- see note in plot_panel


def plot_panel(ax, series_data: list, colors: list, value_key: str, ylabel: str,
               show_background: bool, show_trend: bool):
    legend_handles = []
    legend_labels = []

    for i, sd in enumerate(series_data):
        color = colors[i % len(colors)]
        marker = MARKER_A if sd['house'] == 'A' else MARKER_B

        if show_background and sd['block_values']:
            bt = pd.to_datetime(sd['block_times'])
            bv_key = 'block_values' if value_key == 'value' else 'block_temps'
            bv = np.asarray(sd[bv_key], dtype=float)
            if len(bv) == len(bt):
                ax.scatter(bt, bv, s=3, alpha=0.35, color=color, marker='.',
                           edgecolors='none', zorder=1, rasterized=True)

        files = sd['files']
        ft = [fp['time'] for fp in files]
        fv = [fp[value_key] for fp in files]
        if sd['excluded']:
            h = ax.scatter(ft, fv, s=90, facecolors='none', edgecolors=color,
                           linewidths=1.6, marker=marker, zorder=5)
        else:
            h = ax.scatter(ft, fv, s=70, color=color, edgecolors='black',
                           linewidths=0.8, marker=marker, zorder=5)

        label = sd['name'] + (' (excluded)' if sd['excluded'] else '')
        legend_handles.append(Line2D([0], [0], marker=marker, linestyle='none',
                                      markersize=7,
                                      markerfacecolor=('none' if sd['excluded'] else color),
                                      markeredgecolor=color, markeredgewidth=1.4))
        legend_labels.append(label)

    if show_trend:
        for house, trend_color, label in [('A', TREND_COLOR_A, 'Trend House A (LOWESS)'),
                                           ('B', TREND_COLOR_B, 'Trend House B (LOWESS)')]:
            x_eval, y_eval = house_trend(series_data, house, value_key)
            if x_eval is None:
                continue
            # x_eval is numeric mdates date-nums -- do NOT convert via
            # mdates.num2date here. That returns tz-aware datetimes, and
            # mixing them with the tz-naive pandas Timestamps used above
            # corrupts Axes autoscale (silently clips other artists).
            (h,) = ax.plot(x_eval, y_eval, color=trend_color, linewidth=2.6, zorder=6)
            legend_handles.append(h)
            legend_labels.append(label)

    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    return legend_handles, legend_labels


def main():
    if os.getenv('DOCKER_ENV'):
        data_root = Path('/app/data')
        out_dir = Path('/app/figures')
    else:
        root = Path(__file__).resolve().parent.parent
        data_root = root / 'data'
        out_dir = root / 'figures'

    processed_root = data_root / 'processed'

    print('Collecting per-series NoFlow-equilibrium block + file data...')
    series_data = collect_series(processed_root)
    if not series_data:
        print('No data found under', processed_root, '— nothing to plot.')
        return
    n_files = sum(len(sd['files']) for sd in series_data)
    print(f'{len(series_data)} series, {n_files} files total.')

    colors = series_palette(len(SERIES))

    plt.rcParams.update({
        'font.size': 10,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#1a1a1a',
        'text.color': '#1a1a1a',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), sharex=True, constrained_layout=True,
        facecolor='white',
    )
    ax_top.set_facecolor('white')
    ax_bot.set_facecolor('white')

    handles, labels = plot_panel(
        ax_top, series_data, colors, 'value',
        r'NoFlow baseline $t_{\mathrm{up},0}$ (ps)',
        show_background=True, show_trend=True,
    )
    plot_panel(
        ax_bot, series_data, colors, 'temp', 'Ambient temperature (°C)',
        show_background=False, show_trend=False,
    )

    ax_bot.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax_bot.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    ax_bot.set_xlabel('Recording date', fontsize=10.5)

    ncol = math.ceil(len(handles) / 3)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, fontsize=6.6, frameon=False, handletextpad=0.4,
               columnspacing=1.0, borderaxespad=0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [('pdf', None), ('png', 300)]:
        p = out_dir / f'dataset_drift_overview.{ext}'
        kw = dict(format=ext, bbox_inches='tight', facecolor='white')
        if dpi:
            kw['dpi'] = dpi
        fig.savefig(p, **kw)
        print(f'Saved -> {p}')

    plt.close(fig)


if __name__ == '__main__':
    main()
