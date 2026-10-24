"""
Print a patch-count summary table for Transformer Experiments 3.1, 3.2, 3.3.

For each experiment, prints per-series counts (both train and test):
  - total patches
  - total data points  (patches × patch_width × downsample_factor)
  - noflow patches
  - leakage patches

Usage (from project root, inside Docker or with local env):
    python scripts/print_experiment_patches.py
    python scripts/print_experiment_patches.py --data-path /path/to/data
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

DOWNSAMPLE_FACTOR = 16  # 16 raw samples per downsampled block

EXPERIMENTS = {
    "3.1": {
        "train": [
            "10.01.2026", "24.01.2026", "7.02.2026",
            "7.03.2026", "15.03.2026", "5.04.2026", "11.04.2026",
        ],
        "test": [
            "21.02.2026", "18.04.2026", "ha-25.04.2026", "ha-2.05.2026",
        ],
    },
    "3.2": {
        "train": [
            "10.01.2026", "24.01.2026", "7.02.2026", "21.02.2026",
            "7.03.2026", "15.03.2026", "5.04.2026", "11.04.2026",
            "18.04.2026", "ha-25.04.2026",
        ],
        "test": ["ha-2.05.2026"],
    },
    "3.3": {
        "train": [
            "10.01.2026", "24.01.2026", "7.02.2026", "21.02.2026",
            "7.03.2026", "15.03.2026", "5.04.2026", "11.04.2026",
            "18.04.2026", "ha-2.05.2026",
        ],
        "test": ["ha-25.04.2026"],
    },
}


def load_metadata(data_path: Path, data_root: str = "prepared"):
    base = data_path / data_root
    with open(base / "splits.json") as f:
        splits = json.load(f)
    with open(base / "prepared" / "metadata.json") as f:
        meta_list = json.load(f)
    meta_by_id = {m["segment_id"]: m for m in meta_list}
    return splits, meta_by_id


def series_counts(meta_by_id, series_names):
    """Return {series: {total, noflow, leakage, patch_width}} summed over all segments."""
    counts = defaultdict(lambda: {"total": 0, "noflow": 0, "leakage": 0, "patch_width": 64})
    allowed = set(series_names)
    for meta in meta_by_id.values():
        s = meta.get("series", "")
        if s not in allowed:
            continue
        n = meta.get("n_patches", 0)
        label = meta.get("label", -1)
        pw = meta.get("patch_width", 64)
        counts[s]["total"] += n
        counts[s]["patch_width"] = pw
        if label == 0:
            counts[s]["noflow"] += n
        elif label == 1:
            counts[s]["leakage"] += n
    return counts


def print_experiment(exp_id, exp, meta_by_id):
    print(f"\n{'='*80}")
    print(f"Experiment {exp_id}")
    print(f"{'='*80}")

    for role in ("train", "test"):
        series_list = exp[role]
        counts = series_counts(meta_by_id, series_list)

        header = f"  {role.upper()} series  ({len(series_list)} series)"
        print(f"\n{header}")
        print(f"  {'Series':<22} {'Patches':>9} {'DataPoints':>11} {'NoFlow':>9} {'Leakage':>9}")
        print(f"  {'-'*22} {'-'*9} {'-'*11} {'-'*9} {'-'*9}")

        tot_patches = tot_noflow = tot_leak = 0
        for s in series_list:
            c = counts.get(s)
            if c is None:
                print(f"  {s:<22} {'N/A':>9} {'N/A':>11} {'N/A':>9} {'N/A':>9}")
                continue
            pw = c["patch_width"]
            dp = c["total"] * pw * DOWNSAMPLE_FACTOR
            print(
                f"  {s:<22} {c['total']:>9,} {dp:>11,} {c['noflow']:>9,} {c['leakage']:>9,}"
            )
            tot_patches += c["total"]
            tot_noflow += c["noflow"]
            tot_leak += c["leakage"]

        dp_total = tot_patches * pw * DOWNSAMPLE_FACTOR
        print(f"  {'TOTAL':<22} {tot_patches:>9,} {dp_total:>11,} {tot_noflow:>9,} {tot_leak:>9,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data",
                        help="Base data directory (default: data)")
    parser.add_argument("--data-root", default="prepared",
                        help="Subdirectory within data-path (default: prepared)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_path = root / args.data_path

    print(f"Loading metadata from: {data_path / args.data_root}")
    _, meta_by_id = load_metadata(data_path, args.data_root)
    print(f"  Loaded {len(meta_by_id)} segments")

    for exp_id, exp in EXPERIMENTS.items():
        print_experiment(exp_id, exp, meta_by_id)

    print()


if __name__ == "__main__":
    main()
