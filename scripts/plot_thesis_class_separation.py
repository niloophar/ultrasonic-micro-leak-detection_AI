#!/usr/bin/env python
"""
Thesis Plot 2 — NoFlow vs Leakage Class Separation at Different Flow Rates.

Shows how well NoFlow and leakage can be separated at 100 mL/min (10.01.2026)
versus 20 mL/min (15.03.2026).

Each point in the scatter is a 500-row rolling window mean, which removes
per-measurement noise and reveals the systematic flow signal.

  x-axis: mean t_up_0 (absolute transit time — shows temperature level)
  y-axis: mean tdiff  (upstream minus downstream — shows flow asymmetry)

During NoFlow: tdiff ≈ 0. During leakage: tdiff shifts systematically.
The shift is large at 100 mL/min and barely visible at 20 mL/min.

Output: images/thesis/class_separation.png

Run inside Docker:
    docker compose run --rm kild python scripts/plot_thesis_class_separation.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERIES = {
    '10.01.2026': {'label': 'Jan 10  —  100 mL/min', 'color_nf': '#2166ac', 'color_fl': '#d6604d'},
    '15.03.2026': {'label': 'Mar 15  —  20 mL/min',  'color_nf': '#1a9850', 'color_fl': '#f46d43'},
}
WINDOW  = 500   # rows to average (~1–2 minutes of data)
STEP    = 250   # stride between windows
MAX_PTS = 3000  # max scatter points per class


def load_series(processed_dir: Path, series: str) -> pd.DataFrame:
    series_dir = processed_dir / series
    if not series_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(series_dir.glob('*.csv')):
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def rolling_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return one averaged row per STEP rows using a rolling WINDOW."""
    df = df[df['valid'] == 1].reset_index(drop=True)
    df = df[df['flow_type'].isin(['noflow', 'flow'])].reset_index(drop=True)
    required = ['t_up_0 in ps', 'tdiff in ps', 'flow_type']
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    rows = []
    for start in range(0, len(df) - WINDOW, STEP):
        chunk = df.iloc[start:start + WINDOW]
        majority = chunk['flow_type'].mode().iloc[0]
        rows.append({
            'mean_tup':   chunk['t_up_0 in ps'].mean(),
            'mean_tdiff': chunk['tdiff in ps'].mean(),
            'flow_type':  majority,
        })
    return pd.DataFrame(rows)


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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Leakage signal visibility at high vs low flow rate\n'
        '(each point = 500-measurement window mean)',
        fontsize=12
    )

    rng = np.random.default_rng(42)

    for ax, (series, meta) in zip(axes, SERIES.items()):
        df_raw = load_series(processed_dir, series)
        if df_raw.empty:
            ax.set_title(f'{meta["label"]}\n[no data]')
            continue

        agg = rolling_aggregate(df_raw)
        if agg.empty:
            ax.set_title(f'{meta["label"]}\n[aggregation failed]')
            continue

        nf = agg[agg['flow_type'] == 'noflow']
        fl = agg[agg['flow_type'] == 'flow']

        def subsample(df, n=MAX_PTS):
            if len(df) > n:
                return df.sample(n, random_state=42)
            return df

        nf = subsample(nf)
        fl = subsample(fl)

        ax.scatter(nf['mean_tup'] / 1e6, nf['mean_tdiff'],
                   c=meta['color_nf'], alpha=0.35, s=8,
                   label=f'NoFlow  ({len(nf):,} windows)', rasterized=True)
        ax.scatter(fl['mean_tup'] / 1e6, fl['mean_tdiff'],
                   c=meta['color_fl'], alpha=0.35, s=8,
                   label=f'Leakage ({len(fl):,} windows)', rasterized=True)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Mean transit time $t_{up,0}$ (μs)', fontsize=11)
        ax.set_ylabel('Mean transit time difference\n$t_{diff}$ (ps)', fontsize=11)
        ax.set_title(meta['label'], fontsize=11)
        ax.legend(fontsize=9, markerscale=3)
        ax.grid(linestyle='--', alpha=0.3)

        print(f'{series}: NoFlow windows={len(nf)}, Flow windows={len(fl)}')
        if not fl.empty and not nf.empty:
            sep = abs(fl['mean_tdiff'].mean() - nf['mean_tdiff'].mean())
            noise = nf['mean_tdiff'].std()
            print(f'  mean tdiff: NoFlow={nf["mean_tdiff"].mean():.1f}, '
                  f'Flow={fl["mean_tdiff"].mean():.1f}, SNR≈{sep/noise:.1f}')

    fig.tight_layout()
    out_path = out_dir / 'class_separation.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
