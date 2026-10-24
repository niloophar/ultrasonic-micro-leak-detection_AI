#!/usr/bin/env python
"""
Ground-truth audit of preprocessing + preparation status per series.

Does NOT trust the `processed` flags in configs/preprocessing/*.json (they can
go stale if a server-side run updates the flag but the config edit is never
committed back to git). Instead checks actual files on disk:

- data/processed/<series>/*.csv — non-empty, has a time-like column
- data/prepared/segments/metadata.json — per-series segment counts
- cross-references against configs/preprocessing/label.json's series list
  and configs/contrastive/encoder.json's test_series/loso_series/chrono_series

Run this ON THE SERVER (where the real data lives) — a local checkout only
has a handful of series' data/processed/ folders, if any.

Run: PYTHONPATH=. python scripts/audit_preprocessing_status.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / 'configs' / 'preprocessing' / 'label.json') as f:
        label_cfg = json.load(f)
    expected_series = [s['name'] for s in label_cfg.get('series', [])]

    processed_flags = {}
    for stage in ('label', 'clean', 'relaxation', 'equilibrium'):
        path = ROOT / 'configs' / 'preprocessing' / f'{stage}.json'
        with open(path) as f:
            cfg = json.load(f)
        processed_flags[stage] = {s['name']: s.get('processed') for s in cfg.get('series', [])}

    try:
        with open(ROOT / 'configs' / 'contrastive' / 'encoder.json') as f:
            enc_cfg = json.load(f)
        enc = enc_cfg.get('contrastive', {}).get('encoder', enc_cfg)
        test_series = set(enc.get('test_series') or [])
        loso_series = set(enc.get('loso_series') or [])
        chrono_series = set(enc.get('chrono_series') or [])
    except FileNotFoundError:
        test_series = loso_series = chrono_series = set()

    seg_meta_path = ROOT / 'data' / 'prepared' / 'segments' / 'metadata.json'
    seg_counts = {}
    if seg_meta_path.exists():
        with open(seg_meta_path) as f:
            seg_meta = json.load(f)
        for seg in (seg_meta if isinstance(seg_meta, list) else seg_meta.values()):
            s = seg.get('series') if isinstance(seg, dict) else None
            if s:
                seg_counts[s] = seg_counts.get(s, 0) + 1

    print(f"{'Series':<18} {'processed/data/prepared/label/clean/relax/equil':<12} "
          f"{'CSV files':<10} {'0-byte/empty':<13} {'Segments':<9} {'in test/loso/chrono'}")
    print('-' * 120)

    any_problem = False
    for series in expected_series:
        # The 10 oldest series were renamed to an "ha-" prefix in label.json
        # AFTER their data/processed/ folders were created under the old
        # unprefixed name (documented landmine in CLAUDE.md) — fall back to
        # the unprefixed name if the prefixed path doesn't exist.
        unprefixed = series.split('-', 1)[1] if series.startswith(('ha-', 'hb-')) else series
        candidates = [series] + ([unprefixed] if unprefixed != series else [])
        proc_dir, used_name = None, series
        for cand in candidates:
            d = ROOT / 'data' / 'processed' / cand
            if d.exists():
                proc_dir, used_name = d, cand
                break
        csv_files = sorted(proc_dir.glob('*.csv')) if proc_dir else []
        empty_files = [f.name for f in csv_files if f.stat().st_size < 200]

        flags = ''.join('T' if processed_flags[s].get(series) else 'F'
                         for s in ('label', 'clean', 'relaxation', 'equilibrium'))
        n_seg = max(seg_counts.get(series, 0), seg_counts.get(used_name, 0))
        memberships = ''.join([
            'T' if series in test_series else '.',
            'L' if series in loso_series else '.',
            'C' if series in chrono_series else '.',
        ])

        problem = (not csv_files) or empty_files or (n_seg == 0 and 'C' in memberships)
        marker = ' <-- CHECK' if problem else ''
        if problem:
            any_problem = True
        name_note = f' (on-disk: {used_name})' if used_name != series else ''

        print(f"{series:<18} {flags:<12} {len(csv_files):<10} "
              f"{len(empty_files) if empty_files else '-':<13} {n_seg:<9} {memberships}{marker}{name_note}")

    print('-' * 120)
    print("Flags column = label/clean/relaxation/equilibrium 'processed' value (T/F).")
    print("Memberships = present in encoder.json's test_series(T) / loso_series(L) / chrono_series(C).")
    if any_problem:
        print("\n[ACTION NEEDED] Rows marked CHECK have missing/empty processed CSVs, "
              "or zero extracted segments despite being referenced in chrono_series — "
              "investigate before relying on them for training/eval.")
    else:
        print("\nNo missing/empty files or zero-segment series detected among the expected list.")


if __name__ == '__main__':
    main()
