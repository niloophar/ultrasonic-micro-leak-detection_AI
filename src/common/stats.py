"""
Pipeline statistics computation, printing, and JSON export.

Computes per-series and total statistics after each preprocessing step
by reading the processed CSVs.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_all_csvs(series_dir: Path) -> Optional[pd.DataFrame]:
    """Read and concatenate all CSVs in a series directory."""
    csv_files = sorted(series_dir.glob("*.csv"))
    if not csv_files:
        return None
    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["_file"] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _fmt_pct(value: float) -> str:
    """Format a percentage value."""
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Clean stats
# ---------------------------------------------------------------------------

def compute_clean_stats(
    data_path: Path,
    processed_root: str,
    series_names: List[str],
) -> dict:
    """Compute outlier statistics from the ``valid`` column."""
    per_series = {}
    total_counts = {"inlier": 0, "local_outlier": 0, "global_outlier": 0, "total": 0}

    for name in series_names:
        series_dir = data_path / processed_root / name
        df = _read_all_csvs(series_dir)
        if df is None or "valid" not in df.columns:
            continue
        n = len(df)
        n_inlier = int((df["valid"] == 1).sum())
        n_local = int((df["valid"] == 0).sum())
        n_global = int((df["valid"] == -1).sum())
        per_series[name] = {
            "total_points": n,
            "inlier_pct": round(n_inlier / n * 100, 2) if n else 0,
            "local_outlier_pct": round(n_local / n * 100, 2) if n else 0,
            "global_outlier_pct": round(n_global / n * 100, 2) if n else 0,
            "data_loss_pct": round((n - n_inlier) / n * 100, 2) if n else 0,
        }
        total_counts["inlier"] += n_inlier
        total_counts["local_outlier"] += n_local
        total_counts["global_outlier"] += n_global
        total_counts["total"] += n

    t = total_counts["total"]
    total = {
        "total_points": t,
        "inlier_pct": round(total_counts["inlier"] / t * 100, 2) if t else 0,
        "local_outlier_pct": round(total_counts["local_outlier"] / t * 100, 2) if t else 0,
        "global_outlier_pct": round(total_counts["global_outlier"] / t * 100, 2) if t else 0,
        "data_loss_pct": round((t - total_counts["inlier"]) / t * 100, 2) if t else 0,
    }
    return {"per_series": per_series, "total": total}


# ---------------------------------------------------------------------------
# Relaxation stats
# ---------------------------------------------------------------------------

def compute_relaxation_stats(
    data_path: Path,
    processed_root: str,
    series_names: List[str],
    time_col: str = "Time",
) -> dict:
    """Compute relaxation extraction statistics."""
    per_series = {}
    agg_relaxations: List[dict] = []
    total_valid = 0
    total_valid_relax = 0

    for name in series_names:
        series_dir = data_path / processed_root / name
        df = _read_all_csvs(series_dir)
        if df is None or "relaxation_id" not in df.columns:
            continue

        # Unique relaxation IDs (>= 0)
        relax_ids = df.loc[df["relaxation_id"] >= 0, "relaxation_id"].unique()
        n_relax = len(relax_ids)

        # Per-relaxation stats
        relax_blocks = []
        relax_hours = []
        for rid in relax_ids:
            mask = df["relaxation_id"] == rid
            n_points = int(mask.sum())
            relax_blocks.append(n_points / 64)
            if time_col in df.columns:
                times = pd.to_datetime(df.loc[mask, time_col], errors="coerce")
                times = times.dropna()
                if len(times) >= 2:
                    dur = (times.max() - times.min()).total_seconds() / 3600
                    relax_hours.append(dur)
            agg_relaxations.append({"blocks": n_points / 64, "hours": relax_hours[-1] if relax_hours else 0})

        n_valid = int((df["valid"] == 1).sum()) if "valid" in df.columns else len(df)
        n_valid_relax = int((df["valid_relaxation"] == 1).sum()) if "valid_relaxation" in df.columns else 0

        total_valid += n_valid
        total_valid_relax += n_valid_relax

        per_series[name] = {
            "n_relaxations": n_relax,
            "avg_length_blocks": round(float(np.mean(relax_blocks)), 1) if relax_blocks else 0,
            "avg_length_hours": round(float(np.mean(relax_hours)), 2) if relax_hours else 0,
            "pct_valid_in_relaxations": round(n_valid_relax / n_valid * 100, 2) if n_valid else 0,
        }

    # Total
    all_blocks = [r["blocks"] for r in agg_relaxations]
    all_hours = [r["hours"] for r in agg_relaxations if r["hours"] > 0]
    total = {
        "n_relaxations": len(agg_relaxations),
        "avg_length_blocks": round(float(np.mean(all_blocks)), 1) if all_blocks else 0,
        "avg_length_hours": round(float(np.mean(all_hours)), 2) if all_hours else 0,
        "pct_valid_in_relaxations": round(total_valid_relax / total_valid * 100, 2) if total_valid else 0,
    }
    return {"per_series": per_series, "total": total}


# ---------------------------------------------------------------------------
# Equilibrium stats
# ---------------------------------------------------------------------------

def compute_equilibrium_stats(
    data_path: Path,
    processed_root: str,
    series_names: List[str],
    time_col: str = "Time",
) -> dict:
    """Compute equilibrium segment statistics."""
    per_series = {}
    agg_segments: List[dict] = []
    total_eq_points = 0
    total_relax_points = 0

    for name in series_names:
        series_dir = data_path / processed_root / name
        df = _read_all_csvs(series_dir)
        if df is None or "equilibrium" not in df.columns:
            continue

        # Count contiguous runs of equilibrium == 1
        eq = df["equilibrium"].values
        segments_info = []
        in_seg = False
        seg_start = 0
        for i, v in enumerate(eq):
            if v == 1 and not in_seg:
                in_seg = True
                seg_start = i
            elif v != 1 and in_seg:
                in_seg = False
                segments_info.append((seg_start, i))
        if in_seg:
            segments_info.append((seg_start, len(eq)))

        n_segments = len(segments_info)

        # Per-segment stats
        seg_blocks = []
        seg_minutes = []
        for start, end in segments_info:
            n_points = end - start
            seg_blocks.append(n_points / 64)
            if time_col in df.columns:
                times = pd.to_datetime(df.iloc[start:end][time_col], errors="coerce").dropna()
                if len(times) >= 2:
                    dur = (times.max() - times.min()).total_seconds() / 60
                    seg_minutes.append(dur)
            agg_segments.append({
                "blocks": n_points / 64,
                "minutes": seg_minutes[-1] if seg_minutes else 0,
            })

        n_eq = int((df["equilibrium"] == 1).sum())
        n_relax = int((df["valid_relaxation"] == 1).sum()) if "valid_relaxation" in df.columns else 0

        total_eq_points += n_eq
        total_relax_points += n_relax

        per_series[name] = {
            "n_segments": n_segments,
            "pct_relaxation_in_equilibrium": round(n_eq / n_relax * 100, 2) if n_relax else 0,
            "avg_length_blocks": round(float(np.mean(seg_blocks)), 1) if seg_blocks else 0,
            "avg_length_minutes": round(float(np.mean(seg_minutes)), 1) if seg_minutes else 0,
        }

    # Total
    all_blocks = [s["blocks"] for s in agg_segments]
    all_minutes = [s["minutes"] for s in agg_segments if s["minutes"] > 0]
    total = {
        "n_segments": len(agg_segments),
        "pct_relaxation_in_equilibrium": round(total_eq_points / total_relax_points * 100, 2) if total_relax_points else 0,
        "avg_length_blocks": round(float(np.mean(all_blocks)), 1) if all_blocks else 0,
        "avg_length_minutes": round(float(np.mean(all_minutes)), 1) if all_minutes else 0,
    }
    return {"per_series": per_series, "total": total}


# ---------------------------------------------------------------------------
# Splitting stats
# ---------------------------------------------------------------------------

def _count_patches(n_points: int, downsample_factor: int, patch_width: int, patch_overlap: float) -> int:
    """Compute number of patches a segment yields (same logic as _cut_patches)."""
    n_steps = n_points // downsample_factor
    if n_steps < patch_width:
        return 0
    stride = max(1, int(patch_width * (1.0 - patch_overlap)))
    return (n_steps - patch_width) // stride + 1


def build_splitting_stats(
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    segment_labels: Dict[int, int],
    metadata_by_id: Dict[int, dict],
    downsample_factor: int = 64,
    patch_width: int = 32,
    patch_overlap: float = 0.0,
) -> dict:
    """Build splitting statistics dict from already-computed split data.

    Returns two views:
    - ``per_series``: keyed by series name, each with split role, segment counts,
      leakage/no_leakage breakdown, total points, and patch count.
    - ``per_split``: keyed by split name (train/val/test) with totals.
    - ``total``: grand totals across all splits.
    """
    all_ids = train_ids + val_ids + test_ids
    split_map = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in val_ids:
        split_map[sid] = "val"
    for sid in test_ids:
        split_map[sid] = "test"

    # --- Per-series view (series as rows) ---
    series_data: Dict[str, dict] = {}
    for sid in all_ids:
        s = metadata_by_id[sid]["series"]
        if s not in series_data:
            series_data[s] = {"segments": 0, "leakage": 0, "no_leakage": 0, "points": 0, "patches": 0, "splits": set()}
        series_data[s]["segments"] += 1
        series_data[s]["points"] += metadata_by_id[sid]["n_points"]
        # Overlap only for train segments (augmentation)
        seg_overlap = patch_overlap if split_map[sid] == "train" else 0.0
        series_data[s]["patches"] += _count_patches(
            metadata_by_id[sid]["n_points"], downsample_factor, patch_width, seg_overlap,
        )
        series_data[s]["splits"].add(split_map[sid])
        if segment_labels.get(sid) == 1:
            series_data[s]["leakage"] += 1
        else:
            series_data[s]["no_leakage"] += 1

    per_series = {}
    for name in sorted(series_data):
        d = series_data[name]
        splits = sorted(d["splits"])
        per_series[name] = {
            "split": "/".join(s.capitalize() for s in splits),
            "segments": d["segments"],
            "leakage": d["leakage"],
            "no_leakage": d["no_leakage"],
            "points": d["points"],
            "patches": d["patches"],
        }

    # --- Per-split view (split as rows) ---
    def _split_info(ids: List[int], overlap: float) -> dict:
        n_leak = sum(1 for i in ids if segment_labels.get(i) == 1)
        n_no = sum(1 for i in ids if segment_labels.get(i) == 0)
        n_pts = sum(metadata_by_id[i]["n_points"] for i in ids)
        n_patches = sum(
            _count_patches(metadata_by_id[i]["n_points"], downsample_factor, patch_width, overlap)
            for i in ids
        )
        return {"segments": len(ids), "leakage": n_leak, "no_leakage": n_no, "points": n_pts, "patches": n_patches}

    per_split = {
        "train": _split_info(train_ids, patch_overlap),
        "val": _split_info(val_ids, 0.0),
        "test": _split_info(test_ids, 0.0),
    }

    # --- Total ---
    total = {
        "segments": len(all_ids),
        "leakage": sum(1 for i in all_ids if segment_labels.get(i) == 1),
        "no_leakage": sum(1 for i in all_ids if segment_labels.get(i) == 0),
        "points": sum(metadata_by_id[i]["n_points"] for i in all_ids),
        "patches": per_split["train"]["patches"] + per_split["val"]["patches"] + per_split["test"]["patches"],
    }

    return {
        "per_series": per_series,
        "per_split": per_split,
        "total": total,
        "patch_config": {
            "downsample_factor": downsample_factor,
            "patch_width": patch_width,
            "patch_overlap": patch_overlap,
        },
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_stats_table(stats: dict, step_name: str) -> None:
    """Print a formatted statistics table to the terminal."""
    title = f" {step_name.upper()} STATISTICS "
    width = 72

    print()
    print(f"{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

    if step_name == "clean":
        _print_clean_table(stats, width)
    elif step_name == "extract":
        _print_extract_table(stats, width)
    elif step_name == "equilibrium":
        _print_equilibrium_table(stats, width)
    elif step_name == "splitting":
        _print_splitting_table(stats, width)

    print(f"{'=' * width}")


def _print_clean_table(stats: dict, width: int) -> None:
    header = f"  {'Series':<16} {'Inlier':>8} {'Local':>8} {'Global':>8} {'Data Loss':>10}  {'Points':>12}"
    print(header)
    print(f"  {'-' * (width - 4)}")
    for name, s in sorted(stats["per_series"].items()):
        print(f"  {name:<16} {_fmt_pct(s['inlier_pct']):>8} {_fmt_pct(s['local_outlier_pct']):>8} "
              f"{_fmt_pct(s['global_outlier_pct']):>8} {_fmt_pct(s['data_loss_pct']):>10}  {s['total_points']:>12,}")
    t = stats["total"]
    print(f"  {'-' * (width - 4)}")
    print(f"  {'TOTAL':<16} {_fmt_pct(t['inlier_pct']):>8} {_fmt_pct(t['local_outlier_pct']):>8} "
          f"{_fmt_pct(t['global_outlier_pct']):>8} {_fmt_pct(t['data_loss_pct']):>10}  {t['total_points']:>12,}")


def _print_extract_table(stats: dict, width: int) -> None:
    header = f"  {'Series':<16} {'Relaxations':>12} {'Avg Blocks':>11} {'Avg Hours':>10} {'% Valid in Relax':>17}"
    print(header)
    print(f"  {'-' * (width - 4)}")
    for name, s in sorted(stats["per_series"].items()):
        print(f"  {name:<16} {s['n_relaxations']:>12} {s['avg_length_blocks']:>11.1f} "
              f"{s['avg_length_hours']:>10.2f} {_fmt_pct(s['pct_valid_in_relaxations']):>17}")
    t = stats["total"]
    print(f"  {'-' * (width - 4)}")
    print(f"  {'TOTAL':<16} {t['n_relaxations']:>12} {t['avg_length_blocks']:>11.1f} "
          f"{t['avg_length_hours']:>10.2f} {_fmt_pct(t['pct_valid_in_relaxations']):>17}")


def _print_equilibrium_table(stats: dict, width: int) -> None:
    header = f"  {'Series':<16} {'Segments':>9} {'% Relax in Eq':>14} {'Avg Blocks':>11} {'Avg Minutes':>12}"
    print(header)
    print(f"  {'-' * (width - 4)}")
    for name, s in sorted(stats["per_series"].items()):
        print(f"  {name:<16} {s['n_segments']:>9} {_fmt_pct(s['pct_relaxation_in_equilibrium']):>14} "
              f"{s['avg_length_blocks']:>11.1f} {s['avg_length_minutes']:>12.1f}")
    t = stats["total"]
    print(f"  {'-' * (width - 4)}")
    print(f"  {'TOTAL':<16} {t['n_segments']:>9} {_fmt_pct(t['pct_relaxation_in_equilibrium']):>14} "
          f"{t['avg_length_blocks']:>11.1f} {t['avg_length_minutes']:>12.1f}")


def _print_splitting_table(stats: dict, width: int) -> None:
    pcfg = stats.get("patch_config", {})
    if pcfg:
        print(f"  Patch config: downsample={pcfg.get('downsample_factor')}, "
              f"width={pcfg.get('patch_width')}, overlap={pcfg.get('patch_overlap')}")
        print()

    # Table 1: Per-series overview (like the clean/extract/equilibrium tables)
    print("  Per-Series Overview")
    header = (f"  {'Serie':<16} {'Split':<12} {'Segmente':>9} {'Leakage':>8} "
              f"{'No Leak':>8} {'Patches':>8} {'Punkte':>12}")
    print(header)
    print(f"  {'-' * (width - 4)}")
    for name, s in sorted(stats["per_series"].items()):
        print(f"  {name:<16} {s['split']:<12} {s['segments']:>9} {s['leakage']:>8} "
              f"{s['no_leakage']:>8} {s['patches']:>8} {s['points']:>12,}")
    t = stats["total"]
    print(f"  {'-' * (width - 4)}")
    print(f"  {'TOTAL':<16} {'':<12} {t['segments']:>9} {t['leakage']:>8} "
          f"{t['no_leakage']:>8} {t['patches']:>8} {t['points']:>12,}")

    # Table 2: Per-split summary
    print()
    print("  Per-Split Summary")
    header2 = (f"  {'Split':<12} {'Segmente':>9} {'Leakage':>8} {'No Leak':>8} "
               f"{'Patches':>8} {'Punkte':>12}")
    print(header2)
    print(f"  {'-' * (width - 4)}")
    for split_name in ["train", "val", "test"]:
        s = stats["per_split"][split_name]
        print(f"  {split_name.upper():<12} {s['segments']:>9} {s['leakage']:>8} "
              f"{s['no_leakage']:>8} {s['patches']:>8} {s['points']:>12,}")
    print(f"  {'-' * (width - 4)}")
    print(f"  {'TOTAL':<12} {t['segments']:>9} {t['leakage']:>8} "
          f"{t['no_leakage']:>8} {t['patches']:>8} {t['points']:>12,}")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_stats(stats: dict, output_path: Path) -> None:
    """Save statistics dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {output_path}")
