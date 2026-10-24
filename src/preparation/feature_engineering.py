"""
Feature Engineering Module - Step 2 of Classification Pipeline.

Computes 33 engineered features from raw segment data and saves
per-segment feature CSVs. Optionally plots flow vs noflow histograms
per feature.

Feature modes:
  all (33): 9 interval_up + 9 interval_down + period_sum_up + period_sum_down
            + period_asymmetry + 10 runtime_diff + total_runtime_diff
            + normalized_period_asymmetry
  compact (21): 9 interval_up + 9 interval_down + period_asymmetry
                + total_runtime_diff + normalized_period_asymmetry
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import (
    compute_features,
    get_feature_columns,
    fit_temperature_correction,
    apply_temperature_correction,
)


def _collect_feature_distributions(
    segments_dir: Path,
    series_name: str,
    metadata: List[dict],
    feature_selection: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Collect raw (pre-downsample) feature values per class for histogram plotting.

    Args:
        segments_dir: Path to the segments directory.
        series_name: Name of the series.
        metadata: List of segment metadata dicts for this series.
        feature_selection: 'all' or 'compact'.

    Returns:
        Dict mapping feature_name -> {'flow': np.array, 'noflow': np.array}.
    """
    feature_cols = get_feature_columns(feature_selection)
    label_map = {1: 'flow', 0: 'noflow'}

    # Load series CSV once
    series_csv = segments_dir / f"{series_name}.csv"
    if not series_csv.exists():
        return {}

    series_df = pd.read_csv(series_csv, index_col=False)
    if len(series_df) == 0:
        return {}

    # feature_name -> class_label -> list of arrays
    accum: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for meta in metadata:
        seg_id = meta['segment_id']
        seg_data = series_df[series_df['segment_id'] == seg_id]
        if len(seg_data) == 0:
            continue

        features_df = compute_features(seg_data.reset_index(drop=True))
        available = [c for c in feature_cols if c in features_df.columns]
        if not available:
            continue

        cls = label_map.get(meta['label'])
        if cls is None:
            continue

        for col in available:
            vals = features_df[col].values
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            accum[col][cls].append(vals)

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for feat_name, cls_dict in accum.items():
        result[feat_name] = {}
        for cls, arrays in cls_dict.items():
            result[feat_name][cls] = np.concatenate(arrays) if arrays else np.array([])

    return result


def _generate_segment_drift_plots(
    drift_cfg: dict,
    features_base: Path,
    metadata: List[dict],
    all_feature_cols: List[str],
    img_path: Path,
) -> None:
    """Generate one drift plot per feature from segment-level feature data.

    If drift_cfg['all_features'] is True, plots all columns from the feature
    CSVs (except segment_id). Otherwise uses drift_cfg['features'] list.
    """
    import pandas as pd
    from ..common.plotting import plot_segment_feature_drift

    fit_degree = drift_cfg.get('fit_degree', 3)

    # Determine which features to plot
    if drift_cfg.get('all_features', False):
        features = list(all_feature_cols)
    else:
        features = drift_cfg.get('features', [])

    if not features:
        print("  No features selected — skipping.")
        return

    # Group metadata by series
    series_names = sorted(set(m['series'] for m in metadata))
    meta_by_series: Dict[str, List[dict]] = {s: [] for s in series_names}
    for m in metadata:
        meta_by_series[m['series']].append(m)

    # Load feature CSVs per series
    series_dfs: Dict[str, pd.DataFrame] = {}
    for series_name in series_names:
        csv_path = features_base / f"{series_name}.csv"
        if csv_path.exists():
            series_dfs[series_name] = pd.read_csv(csv_path)

    if not series_dfs:
        print("  No feature CSVs found — skipping drift plots.")
        return

    # Discover all available columns if all_features requested.
    # Skip 'segment_id' and non-numeric passthrough columns like 'Time' (added
    # for the NoFlow time-of-day filter, not a plottable feature).
    if drift_cfg.get('all_features', False):
        all_cols = set()
        for df in series_dfs.values():
            numeric = df.select_dtypes(include='number').columns
            all_cols.update(c for c in numeric if c != 'segment_id')
        features = sorted(all_cols)

    for feature in features:
        available_in = [s for s, df in series_dfs.items() if feature in df.columns]
        if not available_in:
            continue

        series_plot_data = []
        for series_name in series_names:
            if series_name not in series_dfs:
                continue
            df = series_dfs[series_name]
            if feature not in df.columns or 'segment_id' not in df.columns:
                continue

            metas = meta_by_series[series_name]
            times_all, values_all = [], []
            times_noflow, values_noflow = [], []
            times_flow, values_flow = [], []
            for m in metas:
                seg_id = m['segment_id']
                seg_data = df.loc[df['segment_id'] == seg_id, feature].dropna()
                if seg_data.empty:
                    continue
                try:
                    t_start = pd.Timestamp(m['start_time'])
                    t_end = pd.Timestamp(m['end_time'])
                    t_mid = t_start + (t_end - t_start) / 2
                except Exception:
                    continue
                t_iso = t_mid.isoformat()
                val = float(seg_data.mean())
                times_all.append(t_iso)
                values_all.append(val)
                if m.get('label', 0) == 0:
                    times_noflow.append(t_iso)
                    values_noflow.append(val)
                else:
                    times_flow.append(t_iso)
                    values_flow.append(val)

            if times_all:
                series_plot_data.append({
                    'name': series_name,
                    'times_all': times_all,
                    'values_all': values_all,
                    'times_noflow': times_noflow,
                    'values_noflow': values_noflow,
                    'times_flow': times_flow,
                    'values_flow': values_flow,
                })

        if series_plot_data:
            safe_name = feature.replace(' ', '_').replace('/', '_').replace('°', 'deg')
            save_path = img_path / 'prepared' / 'drift' / f"{safe_name}.png"
            plot_segment_feature_drift(feature, series_plot_data, fit_degree, save_path)
            n_total = sum(len(sd['values_all']) for sd in series_plot_data)
            print(f"  {feature}: {n_total} segments across {len(series_plot_data)} series")


def _collect_feature_drift_data(
    features_base: Path,
    metadata: List[dict],
    feature_cols: List[str],
) -> Dict[str, List[dict]]:
    """
    Collect per-series, per-class mean/std for each feature over time.

    Args:
        features_base: Base path to the features directory.
        metadata: List of segment metadata dicts (with 'series', 'label',
            'segment_id', 'start_time', 'end_time').
        feature_cols: List of feature column names to collect stats for.

    Returns:
        Dict mapping feature_name -> list of series dicts sorted chronologically,
        each with keys: 'name', 'mid_time', 'flow_mean', 'flow_std',
        'noflow_mean', 'noflow_std'.
    """
    label_map = {1: 'flow', 0: 'noflow'}

    # Group metadata by series
    series_groups: Dict[str, List[dict]] = defaultdict(list)
    for meta in metadata:
        series_groups[meta['series']].append(meta)

    # Per series: collect feature values by class
    # series_name -> feature_name -> class -> list of arrays
    series_feat_vals: Dict[str, Dict[str, Dict[str, list]]] = {}
    series_mid_times: Dict[str, pd.Timestamp] = {}

    for series_name, metas in series_groups.items():
        feat_vals: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

        # Compute temporal midpoint from segment timestamps
        all_times = []
        for meta in metas:
            try:
                all_times.append(pd.to_datetime(meta['start_time']))
                all_times.append(pd.to_datetime(meta['end_time']))
            except (KeyError, ValueError):
                pass

        if all_times:
            t_min = min(all_times)
            t_max = max(all_times)
            series_mid_times[series_name] = t_min + (t_max - t_min) / 2
        else:
            continue

        # Load series feature CSV once
        series_csv = features_base / f"{series_name}.csv"
        if not series_csv.exists():
            continue
        series_df = pd.read_csv(series_csv, index_col=False)
        if len(series_df) == 0:
            continue

        for meta in metas:
            cls = label_map.get(meta.get('label'))
            if cls is None:
                continue

            seg_id = meta['segment_id']
            df = series_df[series_df['segment_id'] == seg_id]
            if len(df) == 0:
                continue

            for col in feature_cols:
                if col in df.columns:
                    vals = df[col].values
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    feat_vals[col][cls].append(vals)

        series_feat_vals[series_name] = feat_vals

    # Sort series chronologically
    sorted_series = sorted(series_mid_times.keys(), key=lambda s: series_mid_times[s])

    # Build output: feature_name -> sorted list of series dicts
    result: Dict[str, List[dict]] = defaultdict(list)

    for feat_name in feature_cols:
        for series_name in sorted_series:
            feat_vals = series_feat_vals.get(series_name, {}).get(feat_name, {})

            flow_arr = np.concatenate(feat_vals['flow']) if feat_vals.get('flow') else np.array([])
            noflow_arr = np.concatenate(feat_vals['noflow']) if feat_vals.get('noflow') else np.array([])

            # Skip series that have neither class for this feature
            if len(flow_arr) == 0 and len(noflow_arr) == 0:
                continue

            result[feat_name].append({
                'name': series_name,
                'mid_time': series_mid_times[series_name],
                'flow_mean': float(np.mean(flow_arr)) if len(flow_arr) > 0 else np.nan,
                'flow_std': float(np.std(flow_arr)) if len(flow_arr) > 0 else 0.0,
                'noflow_mean': float(np.mean(noflow_arr)) if len(noflow_arr) > 0 else np.nan,
                'noflow_std': float(np.std(noflow_arr)) if len(noflow_arr) > 0 else 0.0,
            })

    return dict(result)


def _series_subfolder_name(series_names: List[str]) -> str:
    """
    Build a human-readable folder name for a combination of series.

    Joins sorted names with '+'. Falls back to a short hash if the
    joined string exceeds 80 characters.
    """
    sorted_names = sorted(series_names)
    joined = "+".join(sorted_names)
    if len(joined) <= 80:
        return joined
    h = hashlib.md5(joined.encode()).hexdigest()[:8]
    return f"{len(sorted_names)}_series_{h}"


def _generate_overlay_drift_plots(
    drift_cfg: dict,
    tc_cfg: dict,
    features_base: Path,
    metadata: List[dict],
    feature_cols: List[str],
    selected_raw: List[str],
    img_path: Path,
) -> None:
    """Generate overlay drift plots: original (solid) vs _tc (dashed).

    Only generated for features that have a corresponding _tc column.
    """
    from ..common.plotting import plot_segment_feature_drift_overlay

    tc_suffix = tc_cfg.get('suffix', '_tc')
    tc_features = tc_cfg.get('features', [])
    fit_degree = drift_cfg.get('fit_degree', 3)

    if not tc_features:
        tc_features = list(feature_cols)

    # Group metadata by series
    series_names = sorted(set(m['series'] for m in metadata))
    meta_by_series: Dict[str, List[dict]] = {s: [] for s in series_names}
    for m in metadata:
        meta_by_series[m['series']].append(m)

    # Load feature CSVs
    series_dfs: Dict[str, pd.DataFrame] = {}
    for series_name in series_names:
        csv_path = features_base / f"{series_name}.csv"
        if csv_path.exists():
            series_dfs[series_name] = pd.read_csv(csv_path)

    if not series_dfs:
        return

    # Only plot features that have both original and _tc columns
    plottable = []
    for feat in tc_features:
        tc_name = f"{feat}{tc_suffix}"
        if any(feat in df.columns and tc_name in df.columns
               for df in series_dfs.values()):
            plottable.append(feat)

    if not plottable:
        return

    print(f"\n{'=' * 60}")
    print("OVERLAY DRIFT PLOTS (original vs TC)")
    print(f"{'=' * 60}")

    def _collect_series_plot_data(feature: str):
        """Collect per-series plot data for a single feature column."""
        result = []
        for series_name in series_names:
            if series_name not in series_dfs:
                continue
            df = series_dfs[series_name]
            if feature not in df.columns or 'segment_id' not in df.columns:
                continue
            metas = meta_by_series[series_name]
            times_all, values_all = [], []
            times_noflow, values_noflow = [], []
            times_flow, values_flow = [], []
            for m in metas:
                seg_data = df.loc[df['segment_id'] == m['segment_id'], feature].dropna()
                if seg_data.empty:
                    continue
                try:
                    t_start = pd.Timestamp(m['start_time'])
                    t_end = pd.Timestamp(m['end_time'])
                    t_mid = t_start + (t_end - t_start) / 2
                except Exception:
                    continue
                t_iso = t_mid.isoformat()
                val = float(seg_data.mean())
                times_all.append(t_iso)
                values_all.append(val)
                if m.get('label', 0) == 0:
                    times_noflow.append(t_iso)
                    values_noflow.append(val)
                else:
                    times_flow.append(t_iso)
                    values_flow.append(val)
            if times_all:
                result.append({
                    'name': series_name,
                    'times_all': times_all, 'values_all': values_all,
                    'times_noflow': times_noflow, 'values_noflow': values_noflow,
                    'times_flow': times_flow, 'values_flow': values_flow,
                })
        return result

    for feat in plottable:
        tc_name = f"{feat}{tc_suffix}"
        data_orig = _collect_series_plot_data(feat)
        data_tc = _collect_series_plot_data(tc_name)

        if data_orig:
            safe_name = feat.replace(' ', '_').replace('/', '_').replace('\u00b0', 'deg')
            save_path = img_path / 'prepared' / 'drift_overlay' / f"{safe_name}.png"
            plot_segment_feature_drift_overlay(
                feat, data_orig, data_tc, fit_degree, save_path,
            )
            print(f"  {feat}")

    print(f"  Saved {len(plottable)} overlay plots to: "
          f"{img_path / 'prepared' / 'drift_overlay'}")


def process_features(
    config: dict,
    data_path: Path,
    img_path: Optional[Path] = None,
    plot_enabled: bool = False,
    only_files: Optional[List[str]] = None,
    max_workers: int = 1,
) -> Dict:
    """
    Compute engineered features for all segments and save as per-segment CSVs.

    Args:
        config: Full config dict (must contain 'classification').
        data_path: Base data path.
        img_path: Path for saving plots.
        plot_enabled: Whether to create feature histogram plots.
        only_files: Optional (unused).
        max_workers: Number of workers (unused).

    Returns:
        Summary dict with feature counts.
    """
    feat_config = config['preparation']['feature_engineering']
    output_root = feat_config['output_root']
    series_list = feat_config.get('series', [])

    # Determine selected features (new format or backward compat)
    selected_computed = feat_config.get('selected_computed_features', [])
    selected_raw = feat_config.get('selected_raw_features', [])
    feature_selection = feat_config.get('features', 'all')

    if not selected_computed:
        # Backward compat with old "features": "all"/"compact" format
        selected_computed = get_feature_columns(feature_selection)

    if not selected_computed:
        selected_computed = get_feature_columns('all')

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Computed:    {len(selected_computed)} features")
    print(f"Raw:         {len(selected_raw)} features")
    print(f"Series:      {series_list}")
    print("=" * 60)

    # Load segment metadata
    segments_base = data_path / output_root / 'segments'
    meta_path = segments_base / 'metadata.json'
    if not meta_path.exists():
        print(f"[ERROR] Metadata not found: {meta_path}")
        print("Run 'extract' step first.")
        return {}

    with open(meta_path, 'r') as f:
        all_metadata = json.load(f)

    feature_cols = list(selected_computed)
    features_base = data_path / output_root / 'features'

    # Load existing features metadata to support incremental runs
    feat_meta_path = features_base / 'metadata.json'
    existing_feat_metadata = []
    if feat_meta_path.exists():
        with open(feat_meta_path, 'r') as f:
            existing_feat_metadata = json.load(f)

    # Keep metadata for series NOT being processed now
    processed_series = set(series_list)
    kept_feat_metadata = [m for m in existing_feat_metadata if m['series'] not in processed_series]

    features_metadata = []
    total_segments = 0
    total_skipped = 0

    for series_name in series_list:
        series_meta = [m for m in all_metadata if m['series'] == series_name]
        if not series_meta:
            print(f"\n  Series: {series_name} — no segments found, skipping")
            continue

        # Load series segment CSV once
        seg_csv = segments_base / f"{series_name}.csv"
        if not seg_csv.exists():
            print(f"\n  Series: {series_name} — segments CSV not found, skipping")
            continue

        seg_series_df = pd.read_csv(seg_csv, index_col=False)
        if len(seg_series_df) == 0:
            print(f"\n  Series: {series_name} — segments CSV empty, skipping")
            continue

        features_base.mkdir(parents=True, exist_ok=True)

        print(f"\n  Series: {series_name} ({len(series_meta)} segments)")

        series_feat_frames = []

        for meta in tqdm(series_meta, desc="    Computing features", unit="seg"):
            seg_id = meta['segment_id']
            df = seg_series_df[seg_series_df['segment_id'] == seg_id].copy()
            df = df.drop(columns=['segment_id']).reset_index(drop=True)

            if len(df) == 0:
                total_skipped += 1
                continue

            # Compute engineered features
            features_df = compute_features(df)
            available_computed = [c for c in feature_cols if c in features_df.columns]

            # Collect raw features (pass-through from segment data)
            available_raw = [c for c in selected_raw if c in df.columns]

            if not available_computed and not available_raw:
                total_skipped += 1
                continue

            # Build output dataframe
            parts = []
            if available_computed:
                comp_arr = features_df[available_computed].values
                comp_arr = np.nan_to_num(comp_arr, nan=0.0, posinf=0.0, neginf=0.0)
                parts.append(pd.DataFrame(comp_arr, columns=available_computed))
            if available_raw:
                raw_arr = df[available_raw].values
                raw_arr = np.nan_to_num(raw_arr, nan=0.0, posinf=0.0, neginf=0.0)
                parts.append(pd.DataFrame(raw_arr, columns=available_raw))

            feat_out = pd.concat(parts, axis=1) if parts else pd.DataFrame()
            feat_out['segment_id'] = seg_id
            # Pass Time through unconditionally — not a feature, but needed at
            # prep time for the NoFlow time-of-day filter (see
            # src/preparation/data_preparation.py :: _load_and_filter_segment_rows).
            # Dropped before downsampling so it never reaches the model.
            if 'Time' in df.columns:
                feat_out['Time'] = df['Time'].values
            series_feat_frames.append(feat_out)

            # Build metadata entry (copy original, no feature_file needed)
            feat_meta = dict(meta)
            features_metadata.append(feat_meta)
            total_segments += 1

        # Write one feature CSV per series
        if series_feat_frames:
            feat_csv = features_base / f"{series_name}.csv"
            feat_series_df = pd.concat(series_feat_frames, ignore_index=True)
            feat_series_df.to_csv(feat_csv, index=False)
            n_segs = sum(1 for m in features_metadata if m['series'] == series_name)
            print(f"    Saved {feat_csv} ({n_segs} segments, {len(feat_series_df)} rows)")

    # Merge with kept metadata from previous runs and save
    merged_feat_metadata = kept_feat_metadata + features_metadata
    features_base.mkdir(parents=True, exist_ok=True)
    with open(feat_meta_path, 'w') as f:
        json.dump(merged_feat_metadata, f, indent=2)
    print(f"\n  Metadata saved: {feat_meta_path}")

    # --- Temperature Correction ---
    tc_cfg = feat_config.get('temperature_correction', {})
    temp_col = tc_cfg.get('temp_column', 'Temp Pipe in \u00b0C')
    tc_suffix = tc_cfg.get('suffix', '_tc')
    tc_features = tc_cfg.get('features', [])

    if tc_cfg.get('enabled', False) and temp_col in selected_raw:
        print(f"\n{'=' * 60}")
        print("TEMPERATURE CORRECTION")
        print(f"{'=' * 60}")
        print(f"  Reference column: {temp_col}")

        # Determine which features to correct
        if not tc_features:
            tc_target_cols = list(feature_cols)
        else:
            tc_target_cols = [c for c in tc_features if c in feature_cols]

        # Collect NoFlow features + temperatures across all series
        noflow_features: Dict[str, list] = defaultdict(list)
        noflow_temps: list = []

        for series_name in series_list:
            feat_csv = features_base / f"{series_name}.csv"
            if not feat_csv.exists():
                continue
            feat_df = pd.read_csv(feat_csv)
            if temp_col not in feat_df.columns:
                continue
            s_meta = [m for m in merged_feat_metadata
                      if m['series'] == series_name]

            for m in s_meta:
                if m.get('label', -1) != 0:
                    continue
                seg_data = feat_df[feat_df['segment_id'] == m['segment_id']]
                if len(seg_data) == 0:
                    continue
                noflow_temps.append(seg_data[temp_col].values)
                for col in tc_target_cols:
                    if col in seg_data.columns:
                        noflow_features[col].append(seg_data[col].values)

        if noflow_temps:
            all_temps = np.concatenate(noflow_temps)
            all_features = {
                k: np.concatenate(v) for k, v in noflow_features.items()
            }

            coefficients = fit_temperature_correction(all_features, all_temps)

            # Save coefficients for reproducibility
            coeff_path = features_base / 'temperature_correction.json'
            with open(coeff_path, 'w') as f:
                json.dump(coefficients, f, indent=2)
            print(f"  T_ref = {coefficients[list(coefficients.keys())[0]]['T_ref']:.2f} \u00b0C")
            print(f"  Coefficients saved: {coeff_path}")

            # Apply correction to all series CSVs
            for series_name in series_list:
                feat_csv = features_base / f"{series_name}.csv"
                if not feat_csv.exists():
                    continue
                feat_df = pd.read_csv(feat_csv)
                if temp_col not in feat_df.columns:
                    continue
                temps = feat_df[temp_col].values
                feat_df = apply_temperature_correction(
                    feat_df, temps, coefficients, tc_suffix,
                )
                feat_df.to_csv(feat_csv, index=False)

            print(f"  Corrected {len(coefficients)} features (suffix: '{tc_suffix}')")
        else:
            print("  [WARN] No NoFlow data with temperature found — skipping")

    elif tc_cfg.get('enabled', False) and temp_col not in selected_raw:
        print(f"\n  [WARN] Temperature correction enabled but '{temp_col}' "
              f"not in selected_raw — skipping")

    # Plot feature histograms
    if plot_enabled and img_path:
        from ..common.plotting import plot_feature_histogram

        print("\n--- FEATURE HISTOGRAMS ---")

        series_distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for series_name in series_list:
            series_meta = [m for m in all_metadata if m['series'] == series_name]
            if not series_meta:
                continue
            series_distributions[series_name] = _collect_feature_distributions(
                segments_base, series_name, series_meta, feature_selection,
            )

        if series_distributions:
            subfolder = _series_subfolder_name(list(series_distributions.keys()))
            hist_dir = img_path / 'classification' / 'features' / subfolder

            # Gather all feature names across series
            all_features = set()
            for dist in series_distributions.values():
                all_features.update(dist.keys())

            for feat_name in sorted(all_features):
                series_data = {}
                for sname in sorted(series_distributions.keys()):
                    feat_dist = series_distributions[sname].get(feat_name, {})
                    if feat_dist:
                        series_data[sname] = feat_dist

                if series_data:
                    save_path = hist_dir / f"{feat_name}.png"
                    plot_feature_histogram(feat_name, series_data, save_path)

            print(f"  Saved {len(all_features)} histograms to: {hist_dir}")

        # Feature correlation plots per group per series
        from ..common.plotting import plot_feature_group_correlation

        CORRELATION_GROUPS = {
            'interval_up': [f'interval_up_{i}' for i in range(9)],
            'interval_down': [f'interval_down_{i}' for i in range(9)],
            'runtime_diff': [f'runtime_diff_{i}' for i in range(10)],
            'norm_interval_up': [f'norm_interval_up_{i}' for i in range(9)],
            'norm_interval_down': [f'norm_interval_down_{i}' for i in range(9)],
            'norm_interval_diff': [f'norm_interval_diff_{i}' for i in range(9)],
            'transit_times_upstream': [f't_up_{i} in ps' for i in range(10)],
            'transit_times_downstream': [f't_down_{i} in ps' for i in range(10)],
        }

        GROUP_LABELS = {
            'interval_up': 'Interval Up',
            'interval_down': 'Interval Down',
            'runtime_diff': 'Runtime Diff',
            'norm_interval_up': 'Norm Interval Up',
            'norm_interval_down': 'Norm Interval Down',
            'norm_interval_diff': 'Norm Interval Diff',
            'transit_times_upstream': 'Transit Times Upstream',
            'transit_times_downstream': 'Transit Times Downstream',
        }

        print("\n--- FEATURE CORRELATION PLOTS ---")
        subfolder = _series_subfolder_name(series_list)
        corr_dir = img_path / 'classification' / 'feature_correlation' / subfolder
        n_corr_plots = 0

        for series_name in series_list:
            feat_csv = features_base / f"{series_name}.csv"
            if not feat_csv.exists():
                continue
            feat_df = pd.read_csv(feat_csv, index_col=False)
            if len(feat_df) == 0:
                continue

            for group_key, group_cols in CORRELATION_GROUPS.items():
                save = corr_dir / f"{group_key}_{series_name}.png"
                plot_feature_group_correlation(
                    feat_df, group_cols, GROUP_LABELS[group_key],
                    series_name, save,
                )
                n_corr_plots += 1

        print(f"  Saved {n_corr_plots} correlation plots to: {corr_dir}")

        # Feature drift plots
        from ..common.plotting import plot_feature_drift_overview, plot_feature_drift_summary

        all_selected_cols = feature_cols + list(selected_raw)
        drift_data = _collect_feature_drift_data(features_base, features_metadata, all_selected_cols)

        if drift_data:
            subfolder = _series_subfolder_name(series_list)
            drift_dir = img_path / 'classification' / 'feature_drift' / subfolder

            print("\n--- FEATURE DRIFT PLOTS ---")

            for feat_name, sdata in sorted(drift_data.items()):
                plot_feature_drift_overview(feat_name, sdata, drift_dir / f"{feat_name}.png")

            plot_feature_drift_summary(drift_data, drift_dir / "drift_summary.png")

            print(f"  Saved {len(drift_data)} drift plots + summary to: {drift_dir}")

    # --- Segment Drift Plot (independent of --plot flag) ---
    drift_cfg = feat_config.get('drift_plot', {})
    if drift_cfg.get('enabled', False) and img_path is not None:
        print(f"\n{'=' * 60}")
        print("SEGMENT DRIFT PLOTS")
        print(f"{'=' * 60}")

        all_selected_for_drift = feature_cols + list(selected_raw)
        _generate_segment_drift_plots(
            drift_cfg, features_base, features_metadata,
            all_selected_for_drift, img_path,
        )

    # --- Overlay Drift Plots (original vs _tc) ---
    if tc_cfg.get('enabled', False) and img_path is not None:
        _generate_overlay_drift_plots(
            drift_cfg, tc_cfg, features_base,
            merged_feat_metadata, feature_cols, selected_raw, img_path,
        )

    all_selected = feature_cols + list(selected_raw)
    summary = {
        'n_computed_features': len(feature_cols),
        'n_raw_features': len(selected_raw),
        'n_features': len(all_selected),
        'total_segments': total_segments,
        'skipped_segments': total_skipped,
    }

    print(f"\n{'=' * 60}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"  Segments processed: {total_segments}")
    print(f"  Segments skipped:   {total_skipped}")
    print(f"{'=' * 60}")

    return summary
