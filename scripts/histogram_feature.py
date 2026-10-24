"""
Histogram of a single feature per series: leakage vs. no-leakage.

Creates one figure with 4 vertical subplots (one per measurement series),
each showing overlaid step histograms for leakage and no-leakage.

Usage:
    python scripts/histogram_feature.py
    python scripts/histogram_feature.py --feature norm_interval_diff_3
    python scripts/histogram_feature.py --save images/histogram_ni_up_3.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Per-series histogram of a feature")
    parser.add_argument("--feature", type=str, default="norm_interval_up_3",
                        help="Feature column to plot (default: norm_interval_up_3)")
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_path = root / args.data_path
    features_base = data_path / "classification" / "features"

    # Load metadata
    with open(features_base / "metadata.json", "r") as f:
        all_metadata = json.load(f)

    # Collect feature values per series per class
    series_names = sorted(set(m["series"] for m in all_metadata))

    data_by_series = {}
    for series in series_names:
        data_by_series[series] = {"leakage": [], "no_leakage": []}

    # Load each series CSV once and collect values
    for series in series_names:
        series_csv = features_base / f"{series}.csv"
        if not series_csv.exists():
            print(f"[WARN] Series CSV not found: {series_csv}, skipping.")
            continue

        df = pd.read_csv(series_csv, index_col=False)
        if args.feature not in df.columns:
            print(f"[WARN] Column '{args.feature}' not in {series_csv.name}, skipping.")
            continue

        # Build segment_id -> label lookup from metadata
        seg_labels = {m["segment_id"]: m["label"] for m in all_metadata if m["series"] == series}

        for seg_id, label in seg_labels.items():
            seg_data = df[df["segment_id"] == seg_id]
            values = seg_data[args.feature].dropna().values
            if len(values) == 0:
                continue
            key = "leakage" if label == 1 else "no_leakage"
            data_by_series[series][key].append(values)

    # Concatenate per series
    for series in series_names:
        for key in ("leakage", "no_leakage"):
            arrays = data_by_series[series][key]
            data_by_series[series][key] = np.concatenate(arrays) if arrays else np.array([])

    # Plot: 4 vertical subplots
    fig, axes = plt.subplots(len(series_names), 1, figsize=(12, 3.5 * len(series_names)),
                              sharex=True)
    if len(series_names) == 1:
        axes = [axes]

    for ax, series in zip(axes, series_names):
        leak = data_by_series[series]["leakage"]
        no_leak = data_by_series[series]["no_leakage"]

        has_leak = len(leak) > 0
        has_no = len(no_leak) > 0

        if has_no:
            ax.hist(no_leak, bins=args.bins, histtype="step", density=True,
                    linewidth=1.5, color="#2196F3", label=f"No Leakage ({len(no_leak):,} pts)")
        if has_leak:
            ax.hist(leak, bins=args.bins, histtype="step", density=True,
                    linewidth=1.5, color="#F44336", label=f"Leakage ({len(leak):,} pts)")

        ax.set_ylabel("Density")
        ax.set_title(f"{series}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(args.feature)

    fig.suptitle(f"Feature: {args.feature}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
