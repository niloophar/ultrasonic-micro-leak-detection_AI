"""Regenerate drift overview plots from cached block stats (no pipeline rerun needed)."""

import argparse
import json
from pathlib import Path

from src.preprocessing.clean import _load_drift_cache, _compute_drift_data_for_series
from src.common.plotting import (
    plot_drift_overview,
    plot_drift_overview_files,
    plot_drift_overview_subplots,
    plot_drift_overview_files_subplots,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data', help='Base data directory')
    parser.add_argument('--img-root', default='./images', help='Base image directory')
    args = parser.parse_args()

    data_path = Path(args.data_root)
    img_path = Path(args.img_root)

    config_path = Path('configs/preprocessing/clean.json')
    with open(config_path) as f:
        clean_cfg = json.load(f)

    drift_cfg = clean_cfg.get('drift_plot', {})
    block_size = drift_cfg.get('block_size', 512)
    fit_degree = drift_cfg.get('fit_degree', 3)
    value_col = drift_cfg.get('value_col', clean_cfg['value_col'])
    time_col = clean_cfg['time_col']
    processed_root = clean_cfg['processed_root']
    cache_dir = data_path / 'stats' / 'drift'

    series_data = []
    for series in clean_cfg['series']:
        name = series['name']
        cache_file = cache_dir / f'{name}.json'
        cached = _load_drift_cache(cache_file, block_size, value_col)
        if cached is not None and 'temp_amb_values' in cached:
            series_data.append(cached)
            print(f'  {name}: {len(cached["values"])} blocks (from cache)')
        else:
            series_dir = data_path / processed_root / name
            drift_data = _compute_drift_data_for_series(name, series_dir, value_col, time_col, block_size)
            if drift_data is not None:
                with open(cache_file, 'w') as f:
                    json.dump(drift_data, f, indent=2)
                series_data.append(drift_data)
                print(f'  {name}: {len(drift_data["values"])} blocks (computed from CSVs)')
            else:
                print(f'  {name}: skipped (no data)')

    if not series_data:
        print('No series data found.')
        return

    out_dir = img_path / 'clean'
    out_dir.mkdir(parents=True, exist_ok=True)

    for save_path, fn in [
        (out_dir / 'drift_overview.png',              plot_drift_overview),
        (out_dir / 'drift_overview_files.png',         plot_drift_overview_files),
        (out_dir / 'drift_overview_subplots.png',      plot_drift_overview_subplots),
        (out_dir / 'drift_overview_files_subplots.png',plot_drift_overview_files_subplots),
    ]:
        fn(series_data, value_col, fit_degree, save_path)
        print(f'  Saved: {save_path}')


if __name__ == '__main__':
    main()
