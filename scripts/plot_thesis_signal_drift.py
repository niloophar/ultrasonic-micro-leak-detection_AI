#!/usr/bin/env python
"""
Thesis Plot 1 — Raw Signal Drift Across Seasons.

Shows how the absolute transit time of NoFlow equilibrium measurements
drifts continuously from January to June as pipe temperature rises.

Output: images/thesis/signal_drift.png

Run inside Docker:
    docker compose run --rm kild python scripts/plot_thesis_signal_drift.py
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

# All valid Haus A series in chronological order (30.03 excluded)
SERIES_ORDER = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
    '7.03.2026', '15.03.2026', '5.04.2026', '11.04.2026',
    '18.04.2026', 'ha-25.04.2026', 'ha-2.05.2026', 'ha-9.05.2026',
    'ha-17.05.2026', 'ha-24.05.2026', 'ha-6.06.2026',
]

SERIES_LABELS = {
    '10.01.2026': 'Jan 10', '24.01.2026': 'Jan 24',
    '7.02.2026': 'Feb 7',  '21.02.2026': 'Feb 21',
    '7.03.2026': 'Mar 7',  '15.03.2026': 'Mar 15',
    '5.04.2026': 'Apr 5',  '11.04.2026': 'Apr 11',
    '18.04.2026': 'Apr 18','ha-25.04.2026': 'Apr 25',
    'ha-2.05.2026': 'May 2','ha-9.05.2026': 'May 9',
    'ha-17.05.2026': 'May 17','ha-24.05.2026': 'May 24',
    'ha-6.06.2026': 'Jun 6',
}


def load_noflow_equilibrium(processed_dir: Path, series: str) -> pd.DataFrame:
    """Load all valid NoFlow equilibrium rows from a series."""
    series_dir = processed_dir / series
    if not series_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(series_dir.glob('*.csv')):
        try:
            df = pd.read_csv(f, parse_dates=['Time'], low_memory=False)
            mask = (df['flow_type'] == 'noflow') & (df['equilibrium'] == 1) & (df['valid'] == 1)
            dfs.append(df[mask])
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    if os.getenv('DOCKER_ENV'):
        data_path = Path('/app/data')
        img_path = Path('/app/images')
    else:
        root = Path(__file__).resolve().parent.parent
        data_path = root / 'data'
        img_path = root / 'images'

    processed_dir = data_path / 'processed'
    out_dir = img_path / 'thesis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect per-series stats ---
    stats = []
    for series in SERIES_ORDER:
        df = load_noflow_equilibrium(processed_dir, series)
        if df.empty:
            print(f'  [skip] {series} — no data')
            continue
        col = 't_up_0 in ps'
        if col not in df.columns:
            print(f'  [skip] {series} — column missing')
            continue
        vals = df[col].dropna().values
        temp = df['Temp Pipe in °C'].dropna().mean()
        stats.append({
            'series': series,
            'label': SERIES_LABELS[series],
            'median': np.median(vals),
            'q25': np.percentile(vals, 25),
            'q75': np.percentile(vals, 75),
            'temp': temp,
        })
        print(f'  {series}: n={len(vals)}, median={np.median(vals):.0f} ps, T={temp:.1f}°C')

    if not stats:
        print('[error] No data found. Check processed/ directory on server.')
        return

    labels = [s['label'] for s in stats]
    medians = np.array([s['median'] for s in stats])
    q25 = np.array([s['q25'] for s in stats])
    q75 = np.array([s['q75'] for s in stats])
    temps = np.array([s['temp'] for s in stats])
    x = np.arange(len(stats))

    # --- Colour each bar by temperature (blue=cold, red=warm) ---
    norm = plt.Normalize(temps.min(), temps.max())
    cmap = plt.cm.coolwarm
    colors = [cmap(norm(t)) for t in temps]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Boxplot-style: bar at median, error bars = IQR
    bars = ax1.bar(x, medians, color=colors, alpha=0.85, zorder=3)
    ax1.errorbar(x, medians, yerr=[medians - q25, q75 - medians],
                 fmt='none', color='#333333', capsize=4, linewidth=1.2, zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Transit time $t_{up,0}$ (ps)', fontsize=12)
    ax1.set_xlabel('Measurement series', fontsize=12)
    ax1.set_title('Raw NoFlow signal drift across seasons (Jan–Jun 2026)', fontsize=13)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e6:.3f}'))
    ax1.set_ylabel('Transit time $t_{up,0}$ (μs)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    # Secondary axis: temperature
    ax2 = ax1.twinx()
    ax2.plot(x, temps, color='tomato', marker='D', markersize=5,
             linewidth=1.5, linestyle='--', label='Pipe temperature', zorder=5)
    ax2.set_ylabel('Pipe temperature (°C)', fontsize=12, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')

    # Colourbar for temperature
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, pad=0.12, shrink=0.85)
    cbar.set_label('Pipe temperature (°C)', fontsize=10)

    # Dashed vertical line separating winter / spring
    spring_start = next((i for i, s in enumerate(stats) if '03' in s['label'] or 'Mar' in s['label']), None)
    if spring_start:
        ax1.axvline(spring_start - 0.5, color='gray', linestyle=':', linewidth=1.2)
        ax1.text(spring_start - 0.4, ax1.get_ylim()[0], 'Spring →',
                 fontsize=9, color='gray', va='bottom')

    fig.tight_layout()
    out_path = out_dir / 'signal_drift.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
