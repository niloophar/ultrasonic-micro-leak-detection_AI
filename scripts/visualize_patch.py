"""
Visualize example patches as heatmaps.

Creates three figures — one per feature group:
  1. norm_interval_up_i    (i=0..8)
  2. norm_interval_down_i  (i=0..8)
  3. norm_interval_diff_i  (i=0..8)

Each figure shows one leakage and one no-leakage patch side by side.

Usage:
    python scripts/visualize_patch.py
    python scripts/visualize_patch.py --downsample-factor 32 --patch-width 64
    python scripts/visualize_patch.py --save-dir images/patches
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Feature groups to visualize
FEATURE_GROUPS = {
    "norm_interval_up": {
        "columns": [f"norm_interval_up_{i}" for i in range(9)],
        "short_prefix": "ni_up_",
        "title": "Normalized Interval — Upstream",
    },
    "norm_interval_down": {
        "columns": [f"norm_interval_down_{i}" for i in range(9)],
        "short_prefix": "ni_down_",
        "title": "Normalized Interval — Downstream",
    },
    "norm_interval_diff": {
        "columns": [f"norm_interval_diff_{i}" for i in range(9)],
        "short_prefix": "ni_diff_",
        "title": "Normalized Interval — Difference (up - down)",
        "symmetric": True,
    },
}


def downsample_segment(feature_array: np.ndarray, factor: int) -> np.ndarray:
    """Average every `factor` consecutive points."""
    n_points, n_features = feature_array.shape
    n_blocks = n_points // factor
    if n_blocks == 0:
        return np.empty((0, n_features))
    truncated = feature_array[: n_blocks * factor]
    return truncated.reshape(n_blocks, factor, n_features).mean(axis=1)


def cut_patches(downsampled: np.ndarray, patch_width: int, overlap: float = 0.0) -> np.ndarray:
    """Cut into fixed-width patches. Returns (n_patches, n_features, patch_width)."""
    n_steps, n_features = downsampled.shape
    if n_steps < patch_width:
        return np.empty((0, n_features, patch_width))
    stride = max(1, int(patch_width * (1.0 - overlap)))
    starts = list(range(0, n_steps - patch_width + 1, stride))
    patches = np.zeros((len(starts), n_features, patch_width))
    for i, s in enumerate(starts):
        patches[i] = downsampled[s : s + patch_width].T
    return patches


def find_examples(all_metadata, features_base, min_points):
    """Find one leakage and one no-leakage segment with enough data."""
    example_leakage = None
    example_noleakage = None

    # Check that the series CSV exists for each candidate
    for meta in all_metadata:
        if example_leakage is not None and example_noleakage is not None:
            break

        series_csv = features_base / f"{meta['series']}.csv"
        if not series_csv.exists():
            continue
        if meta["n_points"] < min_points:
            continue

        if meta["label"] == 1 and example_leakage is None:
            example_leakage = meta
        elif meta["label"] == 0 and example_noleakage is None:
            example_noleakage = meta

    return example_leakage, example_noleakage


# Cache for loaded series CSVs
_series_csv_cache = {}


def build_patch(meta, features_base, feature_cols, downsample_factor, patch_width,
                overlap, patch_index):
    """Load features, downsample, cut patches, return one patch + info."""
    series = meta["series"]
    if series not in _series_csv_cache:
        _series_csv_cache[series] = pd.read_csv(
            features_base / f"{series}.csv", index_col=False
        )

    series_df = _series_csv_cache[series]
    df = series_df[series_df["segment_id"] == meta["segment_id"]].drop(
        columns=["segment_id"]
    ).reset_index(drop=True)
    keep = [c for c in feature_cols if c in df.columns]
    df = df[keep]

    feature_array = df.values
    downsampled = downsample_segment(feature_array, downsample_factor)
    patches = cut_patches(downsampled, patch_width, overlap)

    idx = min(patch_index, len(patches) - 1)
    return patches[idx], keep, len(patches), len(downsampled)


def plot_feature_group(group_key, group_info, examples, features_base, args, save_dir):
    """Create one figure with leakage / no-leakage side by side for a feature group."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Collect patch data first to determine shared colorscale
    patches_data = []
    for class_name, meta in examples:
        patch, keep, n_patches, n_down = build_patch(
            meta, features_base, group_info["columns"],
            args.downsample_factor, args.patch_width, args.overlap, args.patch_index,
        )
        patches_data.append((class_name, meta, patch, keep, n_patches, n_down))

    # Choose colorscale based on feature group:
    #   diff features oscillate around zero -> symmetric diverging scale
    #   up/down features are all positive (~1/9 each) -> use actual data range
    symmetric = group_info.get("symmetric", False)
    all_values = np.concatenate([p[2].ravel() for p in patches_data])
    if symmetric:
        vmax = np.abs(all_values).max()
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin = all_values.min()
        vmax = all_values.max()
        cmap = "viridis"

    for ax, (class_name, meta, patch, keep, n_patches, n_down) in zip(axes, patches_data):
        idx = min(args.patch_index, n_patches - 1)
        im = ax.imshow(patch, aspect="auto", cmap=cmap, interpolation="nearest",
                        vmin=vmin, vmax=vmax)
        ax.set_xlabel("Time step (averaged)")
        ax.set_ylabel("Feature")
        ax.set_yticks(range(len(keep)))
        short = group_info["short_prefix"]
        ax.set_yticklabels([c.replace(group_key + "_", short) for c in keep], fontsize=9)
        ax.set_title(
            f"{class_name} — Patch {idx}/{n_patches}\n"
            f"Segment {meta['segment_id']} | {meta['series']} | "
            f"{meta['n_points']} pts -> {n_down} steps -> {n_patches} patches",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"{group_info['title']}  "
        f"(downsample={args.downsample_factor}, width={args.patch_width}, "
        f"overlap={args.overlap})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_dir is not None:
        save_path = save_dir / f"patch_{group_key}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize example patches as heatmaps")
    parser.add_argument("--data-path", type=str, default="data",
                        help="Base data directory (default: data)")
    parser.add_argument("--downsample-factor", type=int, default=64)
    parser.add_argument("--patch-width", type=int, default=32)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--patch-index", type=int, default=0,
                        help="Which patch to show from each segment (default: 0 = first)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save figures to this directory instead of showing")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_path = root / args.data_path
    features_base = data_path / "classification" / "features"

    # Load metadata
    meta_path = features_base / "metadata.json"
    with open(meta_path, "r") as f:
        all_metadata = json.load(f)

    # Find example segments
    min_points = args.downsample_factor * args.patch_width
    example_leakage, example_noleakage = find_examples(all_metadata, features_base, min_points)

    if example_leakage is None or example_noleakage is None:
        print("Could not find both a leakage and a no-leakage segment.")
        return

    examples = [
        ("Leakage", example_leakage),
        ("No Leakage", example_noleakage),
    ]

    # Prepare save directory
    save_dir = None
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Create one figure per feature group
    for group_key, group_info in FEATURE_GROUPS.items():
        plot_feature_group(group_key, group_info, examples, features_base, args, save_dir)

    if save_dir is None:
        print("Tip: use --save-dir images/patches to save the figures to disk.")


if __name__ == "__main__":
    main()
