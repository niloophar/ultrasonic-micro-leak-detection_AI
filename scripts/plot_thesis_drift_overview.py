#!/usr/bin/env python
"""
Thesis Plot — Drift Overview Across All Haus A Series.

Replicates the style of images/clean/drift_overview.png:
  - Raw NoFlow valid measurements as scatter, coloured by series
  - Diamond marker at the per-series median (placed at the series centre time)
  - Polynomial trend line through the medians
  - Y-axis zoomed to actual data range so drift is visible

Output: images/thesis/drift_overview.png

Run inside Docker:
    docker compose run --rm kild python scripts/plot_thesis_drift_overview.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# All valid Haus A series in order (30.03 excluded)
SERIES_ORDER = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
    '7.03.2026',  '15.03.2026', '5.04.2026', '11.04.2026',
    '18.04.2026', 'ha-25.04.2026', 'ha-2.05.2026', 'ha-9.05.2026',
    'ha-17.05.2026', 'ha-24.05.2026', 'ha-6.06.2026',
]

SERIES_LABELS = {
    '10.01.2026': 'Jan 10',   '24.01.2026': 'Jan 24',
    '7.02.2026':  'Feb 7',    '21.02.2026': 'Feb 21',
    '7.03.2026':  'Mar 7',    '15.03.2026': 'Mar 15',
    '5.04.2026':  'Apr 5',    '11.04.2026': 'Apr 11',
    '18.04.2026': 'Apr 18',   'ha-25.04.2026': 'Apr 25',
    'ha-2.05.2026': 'May 2',  'ha-9.05.2026': 'May 9',
    'ha-17.05.2026': 'May 17','ha-24.05.2026': 'May 24',
    'ha-6.06.2026': 'Jun 6',
}

SUBSAMPLE = 30   # keep every Nth row (speeds up plotting, keeps shape)


def load_series(processed_dir: Path, series: str) -> pd.DataFrame:
    series_dir = processed_dir / series
    if not series_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(series_dir.glob('*.csv')):
        try:
            df = pd.read_csv(f, parse_dates=['Time'], low_memory=False)
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    if os.getenv('DOCKER_ENV'):
        data_path = Path('/app/data')
        img_path  = Path('/app/images')
    else:
        root      = Path(__file__).resolve().parent.parent
        data_path = root / 'data'
        img_path  = root / 'images'

    processed_dir = data_path / 'processed'
    out_dir = img_path / 'thesis'
    out_dir.mkdir(parents=True, exist_ok=True)

    import datetime

    fig, ax = plt.subplots(figsize=(16, 5))

    # --- Season background shading ---
    winter_end = mdates.date2num(datetime.datetime(2026, 3, 20))
    spring_end = mdates.date2num(datetime.datetime(2026, 6, 30))
    data_start = mdates.date2num(datetime.datetime(2025, 12, 15))
    ax.axvspan(data_start, winter_end, color='#cce4f7', alpha=0.35, zorder=0)
    ax.axvspan(winter_end, spring_end, color='#fde8c8', alpha=0.35, zorder=0)

    medians_t   = []
    medians_y   = []
    medians_lbl = []
    loaded_idx  = []   # indices of series that actually loaded, for colouring

    # plasma: dark blue-purple (Jan) → orange-yellow (Jun) — fully saturated, no white
    n_series = len(SERIES_ORDER)
    cmap = plt.cm.plasma

    for idx, series in enumerate(SERIES_ORDER):
        df = load_series(processed_dir, series)
        if df.empty:
            print(f'  [skip] {series} — no data')
            continue

        required = ['Time', 't_up_0 in ps', 'flow_type', 'valid']
        if not all(c in df.columns for c in required):
            print(f'  [skip] {series} — columns missing')
            continue

        df = df[(df['valid'] == 1) & (df['flow_type'] == 'noflow')].copy()
        if df.empty:
            print(f'  [skip] {series} — no valid NoFlow rows')
            continue

        color = cmap(0.08 + 0.84 * idx / (n_series - 1))  # avoid darkest/lightest ends

        df = df.sort_values('Time').reset_index(drop=True)
        df_sub = df.iloc[::SUBSAMPLE]

        times = mdates.date2num(df_sub['Time'].dt.to_pydatetime())
        vals  = df_sub['t_up_0 in ps'].values / 1e6

        ax.scatter(times, vals, color=color, s=1.5, alpha=0.28,
                   rasterized=True, zorder=2)

        med_t = mdates.date2num(df['Time'].median().to_pydatetime())
        med_y = df['t_up_0 in ps'].median() / 1e6

        # Diamond: black fill, coloured edge — visible on any background
        ax.scatter([med_t], [med_y], color=color, marker='D', s=80,
                   zorder=6, edgecolors='black', linewidths=0.8)

        medians_t.append(med_t)
        medians_y.append(med_y)
        medians_lbl.append(SERIES_LABELS[series])
        loaded_idx.append(idx)

        print(f'  {series}: n={len(df)}, median={med_y:.4f} μs')

    # --- Trend line through medians ---
    if len(medians_t) >= 3:
        mt = np.array(medians_t)
        my = np.array(medians_y)
        t_norm = (mt - mt.mean()) / mt.std()
        coeffs = np.polyfit(t_norm, my, deg=2)
        t_fit  = np.linspace(mt.min(), mt.max(), 400)
        y_fit  = np.polyval(coeffs, (t_fit - mt.mean()) / mt.std())
        ax.plot(mdates.num2date(t_fit), y_fit, '-',
                color='black', linewidth=2.2, zorder=7, alpha=0.75,
                label='Trend (quadratic)', solid_capstyle='round')

    # --- Label diamonds ---
    # NOTE ON FONT SCALE: this figure is drawn at figsize=(16, 5) inches but
    # placed in the thesis at \linewidth (~6.10in on A4 with the report's
    # 3.0/2.5cm margins) -- LaTeX therefore shrinks everything by a factor of
    # ~0.38x. All fontsize values below are pre-multiplied by ~3.32x so the
    # smallest in-figure text (annotation labels) still lands at >=~9pt at
    # final print size, matching the caption font. Do not shrink FIG_W to
    # "fix" this -- that just trades one problem for another (fewer weeks
    # visible on the x-axis); scale the fonts instead, as done here.
    for t, y, lbl in zip(medians_t, medians_y, medians_lbl):
        ax.annotate(lbl, xy=(t, y), xytext=(0, 28),
                    textcoords='offset points', fontsize=25,
                    ha='center', va='bottom', color='#222222',
                    fontweight='medium')

    # --- Season labels (in axis fraction space) ---
    ax.text(0.18, 0.04, 'Winter', transform=ax.transAxes,
            fontsize=33, color='#336699', alpha=0.8, va='bottom')
    ax.text(0.65, 0.04, 'Spring / Summer', transform=ax.transAxes,
            fontsize=33, color='#cc7700', alpha=0.8, va='bottom')

    # --- Axes ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=30)
    ax.tick_params(axis='y', labelsize=30)

    all_y  = np.array(medians_y)
    rng    = all_y.max() - all_y.min()
    ax.set_ylim(all_y.min() - rng * 0.9, all_y.max() + rng * 1.6)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))
    ax.set_ylabel('Transit time  $t_{up,0}$  (μs)', fontsize=40)
    ax.set_xlabel('Time', fontsize=40)
    ax.set_title('NoFlow signal drift across all Haus A measurement series  (Jan – Jun 2026)',
                 fontsize=43, pad=10)
    ax.grid(linestyle='--', alpha=0.2, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # --- Colourbar ---
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(0.08, 0.08 + 0.84)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.7, aspect=25)
    cbar.set_label('Jan → Jun', fontsize=30)
    cbar.set_ticks([])

    ax.legend(fontsize=30, loc='lower left', framealpha=0.85,
              handlelength=1.5)

    fig.tight_layout()
    out_path = out_dir / 'drift_overview.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
