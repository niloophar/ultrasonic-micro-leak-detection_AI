#!/usr/bin/env python
"""
Final thesis figures for the pooled chronological extrapolation study
(--run chrono_extrap_pooled): performance vs. chronological test distance,
no-temperature vs. pipe-temperature, plus a LOSO-interpolation reference.

Does NOT rerun or modify any trained model, GMM fit, threshold, or stored
metric value — reads only the existing result JSON files.

Reads:
  data/results/extrapolation_temperature/results/chrono_pooled_none_results.json
  data/results/extrapolation_temperature/results/chrono_pooled_pipe_results.json
  data/results/interpolation_global_gmm/results/loso_results.json  (LOSO reference)
  data/results/extrapolation_temperature/results/chrono_pooled_{none,pipe}_trainthresh_results.json
    (optional bonus figure, outside this prompt's scope — skipped if absent)

Writes (figures/ dir), each as .png (300 dpi) + .pdf:
  extrapolation_temperature_comparison_thesis
  interpolation_vs_extrapolation_reference_thesis
  extrapolation_temperature_delta_thesis
  threshold_protocol_comparison   (bonus, unrenamed, skipped if trainthresh data absent)

Writes (data/results/extrapolation_temperature/results/ dir):
  extrapolation_plot_data.csv
  extrapolation_figure_captions.txt

Run: python scripts/plot_extrapolation_temperature.py
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / 'data' / 'results' / 'extrapolation_temperature' / 'results'
LOSO_PATH = ROOT / 'data' / 'results' / 'interpolation_global_gmm' / 'results' / 'loso_results.json'
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── colours ───────────────────────────────────────────────────────────────
C_DARK    = '#1a1a1a'
C_GRAY    = '#777777'
C_LGRAY   = '#cccccc'
C_NONE    = '#2166ac'   # No Temperature
C_PIPE    = '#c0392b'   # Pipe Temperature
C_INTERP  = '#2e7d32'   # LOSO interpolation reference
C_IMPROVE = '#2e7d32'
C_WORSEN  = '#c0392b'

LW = 1.5
MS = 6.0
AXIS_LABEL = 'Latest recording included in training'

CAPTION_POOLING = (
    "Each extrapolation point is computed from the pooled patches of all recording series "
    "occurring after the final training date."
)
CAPTION_FIT = (
    "The encoder and Global GMM are fitted exclusively on chronologically earlier recording series."
)
CAPTION_THRESHOLD = (
    "Patch classification uses a recording-specific decision threshold calibrated from the known "
    "NoFlow reference patches of each future recording. The experiment therefore represents "
    "deployment with an initial NoFlow calibration phase."
)

_MONTHS = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ─────────────────────────── date formatting ──────────────────────────────

def _strip_house_prefix(series_name: str) -> str:
    return series_name.split('-', 1)[1] if series_name.startswith(('ha-', 'hb-')) else series_name


def _normalized_date(series_name: str) -> str:
    """'ha-9.05.2026' -> '09.05.2026' (zero-padded, no house prefix)."""
    day, month, year = _strip_house_prefix(series_name).split('.')
    return f"{int(day):02d}.{int(month):02d}.{year}"


def _short_date_label(series_name: str) -> str:
    """'ha-9.05.2026' -> '9 May'."""
    day, month, _year = _strip_house_prefix(series_name).split('.')
    return f"{int(day)} {_MONTHS[int(month)]}"


# ─────────────────────────── data loading ────────────────────────────────

def _split_sort_key(split_key: str) -> int:
    return int(split_key.split('_')[0].split('-')[1])


def load_setting(setting: str) -> dict:
    path = RESULTS_DIR / f'chrono_pooled_{setting}_results.json'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `--run chrono_extrap_pooled` on the server "
            f"and scp the results/ folder back first."
        )
    with open(path) as f:
        data = json.load(f)
    data.pop('_summary', None)
    return data


def load_setting_optional(setting: str, suffix: str):
    """Like `load_setting`, but returns None instead of raising if the file
    doesn't exist yet (used for the training_threshold_only protocol, which
    requires its own GPU rerun — see chrono_extrap_pooled_trainthresh). This
    is a separate addition outside the scope of this prompt."""
    path = RESULTS_DIR / f'chrono_pooled_{setting}{suffix}_results.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    data.pop('_summary', None)
    return data


def all_points_with_duplicates(data: dict):
    """All splits sorted by train end, each flagged with `no_additional_data`
    (True where the newly-added training series contributed zero new
    patches — ha-9.05.2026 has no processed data on disk, so that split is
    numerically identical to the previous one)."""
    keys = sorted(data.keys(), key=_split_sort_key)
    points, prev_signature = [], None
    for k in keys:
        m = data[k]
        if 'roc_auc' not in m:
            continue
        signature = (m.get('gmm_fit_n_noflow'), m.get('n_noflow', 0) + m.get('n_flow', 0))
        no_additional_data = signature == prev_signature
        prev_signature = signature
        points.append({
            'train_end': m['n_train_series'],
            'series_name': m['train_series'][-1],
            'train_series_count': m.get('n_train_series'),
            'test_series_count': m.get('n_test_series'),
            'test_series_range': m.get('test_series_range', []),
            'train_noflow_patches': m.get('gmm_fit_n_noflow'),
            'test_noflow_patches': m.get('n_noflow'),
            'test_flow_patches': m.get('n_flow'),
            'auc': m['roc_auc'],
            'f1': m['f1'],
            'accuracy': m.get('accuracy'),
            'no_additional_data': no_additional_data,
        })
    return points


def series_for(points):
    """Main-figure points: drops the no-additional-data duplicate step."""
    return [p for p in points if not p['no_additional_data']]


def load_loso_reference():
    """Per-series LOSO Global-GMM AUC/F1, restricted to AUC-computable series
    (has roc_auc and at least one Flow patch). Also returns the names of
    excluded (empty/NoFlow-only) folds, reported separately per spec."""
    with open(LOSO_PATH) as f:
        d = json.load(f)
    d.pop('_summary', None)
    included = {k: v for k, v in d.items() if 'roc_auc' in v and v.get('n_flow', 0) > 0}
    excluded = sorted(set(d.keys()) - set(included.keys()))
    aucs = [v['roc_auc'] for v in included.values()]
    f1s = [v['f1'] for v in included.values()]
    return {
        'n_series': len(aucs),
        'mean_auc': statistics.mean(aucs),
        'median_auc': statistics.median(aucs),
        'mean_f1': statistics.mean(f1s),
        'median_f1': statistics.median(f1s),
        'excluded_folds': excluded,
    }


# ─────────────────────────── shared styling helpers ──────────────────────

def _style_axes(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=10.5, color=C_DARK)
    ax.set_ylim(0.0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_LGRAY)
    ax.spines['bottom'].set_color(C_LGRAY)
    ax.tick_params(colors=C_DARK, labelsize=9.5)
    ax.grid(axis='y', color=C_LGRAY, lw=0.5, alpha=0.4, zorder=0)
    ax.axhline(0.5, color=C_GRAY, lw=0.8, ls=':', alpha=0.6, zorder=1)


def _date_xaxis(ax, points):
    ax.set_xticks([p['train_end'] for p in points])
    ax.set_xticklabels([_short_date_label(p['series_name']) for p in points],
                        rotation=40, ha='right', fontsize=8.5)


def _add_caption(fig, lines, y0=0.045):
    text = '\n'.join(lines)
    fig.text(0.5, y0, text, ha='center', va='top', fontsize=8, color=C_GRAY, linespacing=1.5)


def _save(fig, out_stem, bottom_margin):
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.94])
    for ext in ('png', 'pdf'):
        kw = dict(format=ext, bbox_inches='tight')
        if ext == 'png':
            kw['dpi'] = 300
        fig.savefig(FIG_DIR / f'{out_stem}.{ext}', **kw)
        print(f'Saved -> {FIG_DIR / f"{out_stem}.{ext}"}')
    plt.close(fig)


def main():
    captions = {}  # figure_stem -> list of caption lines, written to the companion .txt

    none_all = all_points_with_duplicates(load_setting('none'))
    pipe_all = all_points_with_duplicates(load_setting('pipe'))
    none_points = series_for(none_all)
    pipe_points = series_for(pipe_all)

    none_by_end = {p['train_end']: p for p in none_points}
    pipe_by_end = {p['train_end']: p for p in pipe_points}
    for te, p_none in none_by_end.items():
        p_pipe = pipe_by_end.get(te)
        if p_pipe is not None and p_pipe['test_noflow_patches'] + p_pipe['test_flow_patches'] != \
                p_none['test_noflow_patches'] + p_none['test_flow_patches']:
            print(f"[WARNING] train_end={te}: no-temp and pipe-temp settings scored different "
                  f"pooled test sets for this split.")

    # ─── Figure: main temperature-comparison figure ───────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle('Effect of Pipe-Temperature Conditioning on Chronological Extrapolation',
                 fontsize=12.5, fontweight='bold', color=C_DARK, y=0.97)

    te_none = [p['train_end'] for p in none_points]
    te_pipe = [p['train_end'] for p in pipe_points]
    ax1.plot(te_none, [p['auc'] for p in none_points], marker='o', ms=MS, lw=LW, color=C_NONE,
              label='No Temperature', zorder=3, markeredgecolor='white', markeredgewidth=0.7)
    ax1.plot(te_pipe, [p['auc'] for p in pipe_points], marker='o', ms=MS, lw=LW, color=C_PIPE,
              label='Pipe Temperature', zorder=3, markeredgecolor='white', markeredgewidth=0.7)
    ax2.plot(te_none, [p['f1'] for p in none_points], marker='o', ms=MS, lw=LW, color=C_NONE,
              zorder=3, markeredgecolor='white', markeredgewidth=0.7)
    ax2.plot(te_pipe, [p['f1'] for p in pipe_points], marker='o', ms=MS, lw=LW, color=C_PIPE,
              zorder=3, markeredgecolor='white', markeredgewidth=0.7)

    _style_axes(ax1, 'ROC-AUC')
    _style_axes(ax2, 'F1-score')
    _date_xaxis(ax2, none_points)
    ax2.set_xlabel(AXIS_LABEL, fontsize=10.5, color=C_DARK)
    ax1.legend(loc='lower left', frameon=False, fontsize=9.5)  # one shared legend, not repeated

    comp_caption = [CAPTION_POOLING, CAPTION_FIT, CAPTION_THRESHOLD,
                     'Both settings use identical pooled test sets and identical chronological splits.']
    _add_caption(fig, comp_caption, y0=0.02)
    captions['extrapolation_temperature_comparison_thesis'] = comp_caption
    _save(fig, 'extrapolation_temperature_comparison_thesis', bottom_margin=0.20)

    # ─── Figure: LOSO interpolation reference vs. extrapolation ──────────
    loso = load_loso_reference()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle('LOSO Interpolation Reference versus Chronological Extrapolation',
                 fontsize=12.5, fontweight='bold', color=C_DARK, y=0.97)

    ax1.plot(te_none, [p['auc'] for p in none_points], marker='o', ms=MS, lw=LW, color=C_NONE,
              zorder=3, markeredgecolor='white', markeredgewidth=0.7,
              label='Chronological extrapolation (train past, test future — pooled)')
    ax2.plot(te_none, [p['f1'] for p in none_points], marker='o', ms=MS, lw=LW, color=C_NONE,
              zorder=3, markeredgecolor='white', markeredgewidth=0.7)

    ax1.axhline(loso['median_auc'], color=C_INTERP, lw=1.7, ls='-', zorder=2,
                label=f"LOSO interpolation median ROC-AUC — aggregated reference over "
                      f"{loso['n_series']} folds ({loso['median_auc']:.3f})")
    ax1.axhline(loso['mean_auc'], color=C_INTERP, lw=1.1, ls='--', alpha=0.45, zorder=2,
                label=f"LOSO interpolation mean ROC-AUC — aggregated reference ({loso['mean_auc']:.3f})")
    ax2.axhline(loso['median_f1'], color=C_INTERP, lw=1.7, ls='-', zorder=2,
                label=f"LOSO interpolation median F1 ({loso['median_f1']:.3f})")
    ax2.axhline(loso['mean_f1'], color=C_INTERP, lw=1.1, ls='--', alpha=0.45, zorder=2,
                label=f"LOSO interpolation mean F1 ({loso['mean_f1']:.3f})")

    _style_axes(ax1, 'ROC-AUC')
    _style_axes(ax2, 'F1-score')
    _date_xaxis(ax2, none_points)
    ax2.set_xlabel(AXIS_LABEL, fontsize=10.5, color=C_DARK)
    ax1.legend(loc='lower left', frameon=False, fontsize=8)

    interp_caption = [
        CAPTION_POOLING, CAPTION_FIT, CAPTION_THRESHOLD,
        f"LOSO reference is an aggregated statistic over {loso['n_series']} AUC-computable folds, "
        f"not an individual experiment. Excluded (empty/NoFlow-only) folds: "
        f"{', '.join(loso['excluded_folds']) if loso['excluded_folds'] else 'none'}.",
    ]
    _add_caption(fig, interp_caption, y0=0.02)
    captions['interpolation_vs_extrapolation_reference_thesis'] = interp_caption
    _save(fig, 'interpolation_vs_extrapolation_reference_thesis', bottom_margin=0.22)

    # ─── Figure: paired temperature-effect delta figure ───────────────────
    common_ends = sorted(set(none_by_end) & set(pipe_by_end))
    deltas_auc, deltas_f1, delta_series_names = [], [], []
    for te in common_ends:
        p_none, p_pipe = none_by_end[te], pipe_by_end[te]
        deltas_auc.append(p_pipe['auc'] - p_none['auc'])
        deltas_f1.append(p_pipe['f1'] - p_none['f1'])
        delta_series_names.append(p_none['series_name'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('Change in Extrapolation Performance from Pipe-Temperature Conditioning',
                 fontsize=12.5, fontweight='bold', color=C_DARK, y=0.97)

    x = list(range(len(common_ends)))
    for ax, deltas, ylabel in ((ax1, deltas_auc, 'Delta ROC-AUC'), (ax2, deltas_f1, 'Delta F1-score')):
        colors = [C_IMPROVE if d >= 0 else C_WORSEN for d in deltas]
        ax.bar(x, deltas, color=colors, width=0.55, zorder=3)
        ax.axhline(0.0, color=C_DARK, lw=1.0, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10.5, color=C_DARK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(C_LGRAY)
        ax.spines['bottom'].set_color(C_LGRAY)
        ax.tick_params(colors=C_DARK, labelsize=9.5)
        ax.grid(axis='y', color=C_LGRAY, lw=0.5, alpha=0.4, zorder=0)

    ax2.set_xticks(x)
    ax2.set_xticklabels([_short_date_label(s) for s in delta_series_names], rotation=40, ha='right', fontsize=8.5)
    ax2.set_xlabel(AXIS_LABEL, fontsize=10.5, color=C_DARK)

    delta_caption = [
        CAPTION_POOLING,
        'Positive values indicate pipe-temperature conditioning improved performance over no temperature; '
        'negative values indicate it performed worse.',
        f"Delta ROC-AUC: mean={statistics.mean(deltas_auc):+.4f}, median={statistics.median(deltas_auc):+.4f}  |  "
        f"Delta F1: mean={statistics.mean(deltas_f1):+.4f}, median={statistics.median(deltas_f1):+.4f}",
    ]
    _add_caption(fig, delta_caption, y0=0.04)
    captions['extrapolation_temperature_delta_thesis'] = delta_caption
    _save(fig, 'extrapolation_temperature_delta_thesis', bottom_margin=0.20)

    # ─── extrapolation_plot_data.csv — detailed table, all splits ─────────
    import csv
    csv_path = RESULTS_DIR / 'extrapolation_plot_data.csv'
    fieldnames = [
        'train_end_index', 'latest_training_date', 'train_series_count', 'test_series_count',
        'test_date_range', 'train_noflow_patches', 'test_noflow_patches', 'test_flow_patches',
        'roc_auc_no_temperature', 'f1_no_temperature', 'roc_auc_pipe_temperature', 'f1_pipe_temperature',
        'delta_roc_auc', 'delta_f1', 'no_additional_data',
    ]
    none_by_end_all = {p['train_end']: p for p in none_all}
    pipe_by_end_all = {p['train_end']: p for p in pipe_all}
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for te in sorted(none_by_end_all):
            n = none_by_end_all[te]
            p = pipe_by_end_all.get(te, {})
            test_series = n['test_series_range']
            test_range = f"{_normalized_date(test_series[0])}..{_normalized_date(test_series[-1])}" if test_series else ''
            auc_none, f1_none = n['auc'], n['f1']
            auc_pipe, f1_pipe = p.get('auc'), p.get('f1')
            delta_auc = (auc_pipe - auc_none) if auc_pipe is not None else ''
            delta_f1 = (f1_pipe - f1_none) if f1_pipe is not None else ''
            writer.writerow({
                'train_end_index': te,
                'latest_training_date': _normalized_date(n['series_name']),
                'train_series_count': n['train_series_count'],
                'test_series_count': n['test_series_count'],
                'test_date_range': test_range,
                'train_noflow_patches': n['train_noflow_patches'],
                'test_noflow_patches': n['test_noflow_patches'],
                'test_flow_patches': n['test_flow_patches'],
                'roc_auc_no_temperature': round(auc_none, 4),
                'f1_no_temperature': round(f1_none, 4),
                'roc_auc_pipe_temperature': round(auc_pipe, 4) if auc_pipe is not None else '',
                'f1_pipe_temperature': round(f1_pipe, 4) if f1_pipe is not None else '',
                'delta_roc_auc': round(delta_auc, 4) if delta_auc != '' else '',
                'delta_f1': round(delta_f1, 4) if delta_f1 != '' else '',
                'no_additional_data': n['no_additional_data'],
            })
    print(f'Saved -> {csv_path}')

    # ─── extrapolation_figure_captions.txt — companion caption file ───────
    captions_path = RESULTS_DIR / 'extrapolation_figure_captions.txt'
    with open(captions_path, 'w') as f:
        for stem, lines in captions.items():
            f.write(f"{stem}\n")
            f.write('-' * len(stem) + '\n')
            for line in lines:
                f.write(line + '\n')
            f.write('\n')
    print(f'Saved -> {captions_path}')

    # ─── Bonus (outside this prompt's scope): calibrated vs. training-only
    #     threshold protocol — kept as a separate addition, unrenamed,
    #     skipped entirely if its GPU rerun hasn't produced data yet. ───────
    trainthresh_none_raw = load_setting_optional('none', '_trainthresh')
    if trainthresh_none_raw is None:
        print('[skip] threshold-protocol comparison figure (bonus, outside this prompt) — '
              f'{RESULTS_DIR / "chrono_pooled_none_trainthresh_results.json"} not found')
        return

    tt_points = series_for(all_points_with_duplicates(trainthresh_none_raw))
    tt_by_end = {p['train_end']: p for p in tt_points}
    common = sorted(set(none_by_end) & set(tt_by_end))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle('Calibrated vs. Training-Only Threshold Protocol (No Temperature)',
                 fontsize=12.5, fontweight='bold', color=C_DARK, y=0.97)
    cal_pts = [none_by_end[te] for te in common]
    tt_pts = [tt_by_end[te] for te in common]
    ax1.plot(common, [p['f1'] for p in cal_pts], marker='o', ms=MS, lw=LW, color=C_NONE,
              label='Test-series calibrated', zorder=3, markeredgecolor='white', markeredgewidth=0.7)
    ax1.plot(common, [p['f1'] for p in tt_pts], marker='s', ms=MS, lw=LW, color=C_PIPE,
              label='Training-threshold only', zorder=3, markeredgecolor='white', markeredgewidth=0.7)
    ax2.plot(common, [p['accuracy'] for p in cal_pts], marker='o', ms=MS, lw=LW, color=C_NONE,
              zorder=3, markeredgecolor='white', markeredgewidth=0.7)
    ax2.plot(common, [p['accuracy'] for p in tt_pts], marker='s', ms=MS, lw=LW, color=C_PIPE,
              zorder=3, markeredgecolor='white', markeredgewidth=0.7)

    auc_mismatches = [te for te in common if abs(none_by_end[te]['auc'] - tt_by_end[te]['auc']) > 1e-6]
    if auc_mismatches:
        print(f"[WARNING] ROC-AUC differs between calibrated and training-threshold-only "
              f"protocols for train_end={auc_mismatches} — expected identical (threshold-independent).")

    _style_axes(ax1, 'F1-score')
    _style_axes(ax2, 'Accuracy')
    _date_xaxis(ax2, cal_pts)
    ax2.set_xlabel(AXIS_LABEL, fontsize=10.5, color=C_DARK)
    ax1.legend(loc='upper right', frameon=False, fontsize=9.5)
    _add_caption(fig, [
        CAPTION_POOLING,
        'Calibrated: threshold set per future test series from its own NoFlow reference patches. '
        'Training-only: single threshold fixed from training-NoFlow scores, applied unchanged to '
        'every future test patch (no future-series calibration).',
    ], y0=0.03)
    _save(fig, 'threshold_protocol_comparison', bottom_margin=0.20)


if __name__ == '__main__':
    main()
