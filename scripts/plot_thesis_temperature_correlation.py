#!/usr/bin/env python
"""
Thesis Plot 3 — Temperature vs Raw Signal Correlation Across Series.

For each Haus A series, computes the mean of t_up_0 and mean pipe temperature
from valid NoFlow equilibrium rows, then shows the linear relationship.
Demonstrates that seasonal drift in the raw signal is directly driven by temperature.

Output: images/thesis/temperature_correlation.png

Run inside Docker:
    docker compose run --rm kild python scripts/plot_thesis_temperature_correlation.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERIES_ORDER = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026',
    '7.03.2026', '15.03.2026', '5.04.2026', '11.04.2026',
    '18.04.2026', 'ha-25.04.2026', 'ha-2.05.2026', 'ha-9.05.2026',
    'ha-17.05.2026', 'ha-24.05.2026', 'ha-6.06.2026',
]

SERIES_LABELS = {
    '10.01.2026': 'Jan 10', '24.01.2026': 'Jan 24',
    '7.02.2026': 'Feb 7',   '21.02.2026': 'Feb 21',
    '7.03.2026': 'Mar 7',   '15.03.2026': 'Mar 15',
    '5.04.2026': 'Apr 5',   '11.04.2026': 'Apr 11',
    '18.04.2026': 'Apr 18', 'ha-25.04.2026': 'Apr 25',
    'ha-2.05.2026': 'May 2','ha-9.05.2026': 'May 9',
    'ha-17.05.2026': 'May 17','ha-24.05.2026': 'May 24',
    'ha-6.06.2026': 'Jun 6',
}


def load_noflow_equilibrium(processed_dir: Path, series: str) -> pd.DataFrame:
    series_dir = processed_dir / series
    if not series_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(series_dir.glob('*.csv')):
        try:
            df = pd.read_csv(f, low_memory=False)
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

    points = []
    for series in SERIES_ORDER:
        df = load_noflow_equilibrium(processed_dir, series)
        if df.empty or 't_up_0 in ps' not in df.columns:
            print(f'  [skip] {series}')
            continue
        mean_tup = df['t_up_0 in ps'].mean()
        mean_temp = df['Temp Pipe in °C'].mean()
        points.append({
            'series': series,
            'label': SERIES_LABELS[series],
            'mean_tup': mean_tup,
            'mean_temp': mean_temp,
        })
        print(f'  {series}: mean t_up_0={mean_tup:.0f} ps, T_pipe={mean_temp:.2f}°C')

    if len(points) < 3:
        print('[error] Not enough series found.')
        return

    temps = np.array([p['mean_temp'] for p in points])
    tups  = np.array([p['mean_tup']  for p in points])
    labels = [p['label'] for p in points]

    # Linear regression
    slope, intercept, r, p_val, _ = linregress(temps, tups)
    pearson_r, _ = pearsonr(temps, tups)
    t_fit = np.linspace(temps.min() - 0.3, temps.max() + 0.3, 100)
    y_fit = slope * t_fit + intercept

    # Colour points by chronological order
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(points) - 1, 1)) for i in range(len(points))]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (p, c) in enumerate(zip(points, colors)):
        ax.scatter(p['mean_temp'], p['mean_tup'] / 1e6, color=c, s=80, zorder=4)
        ax.annotate(p['label'], (p['mean_temp'], p['mean_tup'] / 1e6),
                    textcoords='offset points', xytext=(5, 4), fontsize=8, color=c)

    ax.plot(t_fit, (slope * t_fit + intercept) / 1e6,
            'k--', linewidth=1.4, label=f'Linear fit  (r = {pearson_r:.2f})', zorder=3)

    ax.set_xlabel('Mean pipe temperature (°C)', fontsize=12)
    ax.set_ylabel('Mean transit time $t_{up,0}$ (μs)', fontsize=12)
    ax.set_title('Temperature-driven signal drift — NoFlow equilibrium mean per series', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.35)

    # Colourbar for chronological order
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(points) - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Series (Jan → Jun)', fontsize=9)
    cbar.set_ticks([0, len(points) - 1])
    cbar.set_ticklabels(['Jan', 'Jun'])

    fig.tight_layout()
    out_path = out_dir / 'temperature_correlation.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')
    print(f'Pearson r = {pearson_r:.3f}  (slope = {slope:.1f} ps/°C)')


if __name__ == '__main__':
    main()
