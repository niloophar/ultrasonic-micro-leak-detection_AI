#!/usr/bin/env python
"""
Thesis-ready result tables for the pooled chronological extrapolation study
(--run chrono_extrap_pooled / --run chrono_extrap_pooled_trainthresh):
train on the earliest series, test on ALL remaining (future) series pooled
together, repeated for no-temperature and temp_pipe-conditioned scoring,
under two threshold protocols.

Reads (test_series_calibrated protocol — per-series threshold from each
future test series' own NoFlow reference patches; always available):
  data/results/extrapolation_temperature/results/chrono_pooled_none_results.json
  data/results/extrapolation_temperature/results/chrono_pooled_pipe_results.json

Reads (training_threshold_only protocol — single threshold fixed from
training-NoFlow scores only, no future-series calibration; optional, only
included once its GPU rerun has completed):
  data/results/extrapolation_temperature/results/chrono_pooled_none_trainthresh_results.json
  data/results/extrapolation_temperature/results/chrono_pooled_pipe_trainthresh_results.json

Writes (same directory):
  extrapolation_temperature_full.csv      — every metric incl. confusion matrix, both settings x both protocols

This is the protocol-comparison table (req. "compare the two threshold
protocols using ROC-AUC, F1, accuracy, TP, TN, FP, FN") — kept as a
separate addition alongside `scripts/plot_extrapolation_temperature.py`,
which produces the primary thesis figures/table/captions (single
test_series_calibrated protocol only, per the plotting-correction prompt).

Run: python scripts/export_extrapolation_temperature_tables.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'data' / 'results' / 'extrapolation_temperature' / 'results'

SETTINGS = [('none', 'No Temperature'), ('pipe', 'Pipe Temperature')]
PROTOCOLS = [('test_series_calibrated', ''), ('training_threshold_only', '_trainthresh')]


def _split_sort_key(split_key: str) -> int:
    # "train1-4_test5-17" -> 4 (train end count) — sorts chronologically.
    return int(split_key.split('_')[0].split('-')[1])


def load_protocol(protocol_suffix: str):
    """Returns {setting: data} for one protocol, or None if its result files
    don't exist yet (e.g. training_threshold_only before its GPU rerun)."""
    out = {}
    for setting, _ in SETTINGS:
        path = RESULTS_DIR / f'chrono_pooled_{setting}{protocol_suffix}_results.json'
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        data.pop('_summary', None)
        out[setting] = data
    return out


def _flag_no_new_data(data_none: dict, keys: list) -> dict:
    """Per split_key -> True if this split added zero new patches versus the
    previous split (ha-9.05.2026 has no processed data on disk)."""
    flags, prev_signature = {}, None
    for k in keys:
        m = data_none.get(k, {})
        signature = (m.get('gmm_fit_n_noflow'), m.get('n_noflow', 0) + m.get('n_flow', 0))
        flags[k] = signature == prev_signature
        prev_signature = signature
    return flags


def main():
    loaded = {}
    for protocol_label, suffix in PROTOCOLS:
        data = load_protocol(suffix)
        if data is None:
            print(f"[skip] protocol '{protocol_label}' — result files not found "
                  f"(chrono_pooled_{{none,pipe}}{suffix}_results.json)")
            continue
        loaded[protocol_label] = data

    if not loaded:
        raise SystemExit("No protocol result files found — nothing to export.")

    # ─── Full CSV: every metric + confusion matrix, both settings x protocols ───
    full_path = RESULTS_DIR / 'extrapolation_temperature_full.csv'
    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Train End', 'N Train Series', 'Test Range', 'N Test Series', 'Setting', 'Threshold Protocol',
            'ROC-AUC', 'F1', 'Accuracy', 'N Correct', 'N Wrong',
            'TP', 'TN', 'FP', 'FN', 'N NoFlow', 'N Flow', 'GMM Fit N (NoFlow)',
            'Train Series (first..last)', 'Test Series (first..last)', 'No New Data (ha-9.05.2026)',
        ])
        for protocol_label, data in loaded.items():
            split_keys = sorted(data['none'].keys(), key=_split_sort_key)
            no_new_data = _flag_no_new_data(data['none'], split_keys)
            for split_key in split_keys:
                for setting, setting_label in SETTINGS:
                    m = data[setting].get(split_key, {})
                    if not m:
                        continue
                    train_series = m.get('train_series', [])
                    test_series = m.get('test_series_range', [])
                    cm = m.get('confusion_matrix', {})
                    n_train_end = m.get('n_train_series', '')
                    n_test = m.get('n_test_series', '')
                    test_range = f"{n_train_end + 1}-{n_train_end + n_test}" if isinstance(n_train_end, int) and isinstance(n_test, int) else ''
                    writer.writerow([
                        n_train_end, n_train_end, test_range, n_test, setting_label, protocol_label,
                        round(m['roc_auc'], 4) if 'roc_auc' in m else '',
                        round(m['f1'], 4) if 'f1' in m else '',
                        round(m['accuracy'], 4) if 'accuracy' in m else '',
                        m.get('n_correct', ''), m.get('n_wrong', ''),
                        cm.get('tp', ''), cm.get('tn', ''), cm.get('fp', ''), cm.get('fn', ''),
                        m.get('n_noflow', ''), m.get('n_flow', ''), m.get('gmm_fit_n_noflow', ''),
                        f"{train_series[0]}..{train_series[-1]}" if train_series else '',
                        f"{test_series[0]}..{test_series[-1]}" if test_series else '',
                        'yes' if no_new_data.get(split_key) else '',
                    ])
    print(f'Saved -> {full_path}')


if __name__ == '__main__':
    main()
