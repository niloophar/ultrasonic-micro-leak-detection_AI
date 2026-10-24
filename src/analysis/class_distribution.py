"""
Class Distribution Analysis.

Computes and visualizes the class distribution (segment count + patch count)
per measurement series, reading from the segments and prepared metadata files.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def compute_class_distribution(
    data_path: Path,
    output_root: str = './prepared',
) -> Dict:
    """
    Compute per-series class distribution from segment and prepared metadata.

    Args:
        data_path: Base data directory.
        output_root: Relative path to classification output (default: ./prepared).

    Returns:
        Dict with keys:
        - series_order: List of series names sorted chronologically.
        - segment_counts: {series: {flow_type: count}} from segments metadata.
        - patch_counts: {series: {flow_type: count}} from prepared metadata.
    """
    from datetime import datetime

    class_base = data_path / output_root

    # --- Segment counts from segments/metadata.json ---
    seg_meta_path = class_base / 'segments' / 'metadata.json'
    if not seg_meta_path.exists():
        print(f"[WARNING] Segment metadata not found: {seg_meta_path}")
        return {'series_order': [], 'segment_counts': {}, 'patch_counts': {}}

    with open(seg_meta_path, 'r') as f:
        seg_metadata = json.load(f)

    segment_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in seg_metadata:
        series = entry['series']
        flow_type = entry.get('flow_type', 'unknown')
        segment_counts[series][flow_type] += 1

    # --- Patch counts from prepared/metadata.json ---
    prep_meta_path = class_base / 'prepared' / 'metadata.json'
    patch_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if prep_meta_path.exists():
        with open(prep_meta_path, 'r') as f:
            prep_metadata = json.load(f)
        for entry in prep_metadata:
            series = entry['series']
            flow_type = entry.get('flow_type', 'unknown')
            n_patches = entry.get('n_patches', 0)
            patch_counts[series][flow_type] += n_patches
    else:
        print(f"[WARNING] Prepared metadata not found: {prep_meta_path}")
        print("  Patch counts will be empty.")

    # Sort series chronologically
    all_series = sorted(
        set(list(segment_counts.keys()) + list(patch_counts.keys())),
        key=lambda s: _parse_series_date(s),
    )

    return {
        'series_order': all_series,
        'segment_counts': {s: dict(segment_counts[s]) for s in all_series},
        'patch_counts': {s: dict(patch_counts[s]) for s in all_series},
    }


def _parse_series_date(name: str):
    """Parse series name as DD.MM.YYYY for sorting."""
    from datetime import datetime
    try:
        return datetime.strptime(name, "%d.%m.%Y")
    except ValueError:
        from datetime import datetime as dt
        return dt.max


def plot_class_distribution(
    distribution: Dict,
    save_path: Path,
) -> None:
    """
    Plot class distribution as a single figure with dual y-axes.

    Left y-axis: segment counts (bars).
    Right y-axis: patch counts (bars, hatched).

    Args:
        distribution: Output of compute_class_distribution().
        save_path: Path to save the figure (e.g. .../class_distribution.png).
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    series_order = distribution['series_order']
    segment_counts = distribution['segment_counts']
    patch_counts = distribution['patch_counts']

    if not series_order:
        print("[WARNING] No series data to plot.")
        return

    classes = ['flow', 'noflow']
    colors = {'flow': '#E53935', 'noflow': '#1E88E5'}
    labels = {'flow': 'Flow (leakage)', 'noflow': 'No flow'}

    n = len(series_order)
    x = np.arange(n)
    # 4 bars per group: 2 segment + 2 patch
    group_width = 0.75
    bar_width = group_width / 4

    fig, ax_seg = plt.subplots(figsize=(max(8, n * 2.5), 5))
    ax_pat = ax_seg.twinx()

    # Segment bars (solid, left side of each group)
    for i, cls in enumerate(classes):
        vals = [segment_counts.get(s, {}).get(cls, 0) for s in series_order]
        pos = x - group_width / 2 + bar_width * (i + 0.5)
        bars = ax_seg.bar(pos, vals, bar_width, label=f'{labels[cls]} (segments)',
                          color=colors[cls], alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax_seg.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(v), ha='center', va='bottom', fontsize=8, zorder=4)

    # Patch bars (hatched, right side of each group)
    for i, cls in enumerate(classes):
        vals = [patch_counts.get(s, {}).get(cls, 0) for s in series_order]
        pos = x - group_width / 2 + bar_width * (i + 2.5)
        bars = ax_pat.bar(pos, vals, bar_width, label=f'{labels[cls]} (patches)',
                          color=colors[cls], alpha=0.40, hatch='//', edgecolor=colors[cls],
                          zorder=2)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax_pat.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(v), ha='center', va='bottom', fontsize=8,
                            fontstyle='italic', zorder=4)

    ax_seg.set_ylabel('Segment Count')
    ax_pat.set_ylabel('Patch Count')
    ax_seg.set_title('Class Distribution per Series')
    ax_seg.set_xticks(x)
    ax_seg.set_xticklabels(series_order, rotation=15)
    ax_seg.grid(True, alpha=0.3, axis='y', zorder=0)

    # Combined legend from both axes
    h1, l1 = ax_seg.get_legend_handles_labels()
    h2, l2 = ax_pat.get_legend_handles_labels()
    ax_seg.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Class distribution plot saved: {save_path}")
