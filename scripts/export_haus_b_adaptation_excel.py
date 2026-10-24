#!/usr/bin/env python
"""
Export the Haus B adaptation investigation to a single formatted Excel
workbook — including the corrected ("fair") per-site evaluation that
replaced the earlier leaky one, and the Part 1 DeepSAD center diagnostic.

IMPORTANT (2026-07-21): the per-site "recalibration works" story from
earlier in this investigation used a leaky protocol (_evaluate_on_test)
that fit the GMM on a series' NoFlow and then scored that SAME NoFlow again.
The corrected ("fair") protocol splits each series' NoFlow chronologically
into disjoint calibration/test files. The fair result is dramatically worse
(mean AUC 0.945 -> ~0.31) — see the "Per-Site FAIR" sheets, now the
headline, and "Per-Site OPTIMISTIC_OLD" kept only for comparison, clearly
flagged as invalid.

Reads:
  data/results/haus_b_adaptation/deepsad_center_check.json
    (Part 1 — pooled two-house DeepSAD center diagnostic)
  data/contrastive_g5_deepsad_transformer_ae_allha/results/test_metrics.json
    (Part 2 — FAIR/corrected per-site eval, current headline)
  data/contrastive_g5_deepsad_transformer_ae_allha/results/test_metrics_fair_folds.json
    (Part 2 — full per-fold audit trail: calibration/test/leak files, date ranges, gmm_mode)
  data/contrastive_g5_deepsad_transformer_ae_allha/results/majority_vote_results.json
    (Part 2 — FAIR per-file majority vote)
  data/contrastive_g5_deepsad_transformer_ae_allha/results/test_metrics_OPTIMISTIC_OLD.json
    (Part 2 — OLD leaky per-site eval, kept only for side-by-side comparison)
  data/contrastive_g5_deepsad_transformer_ae_allha/results/test_metrics_global_hb_extrap.json
  data/contrastive_g5_deepsad_transformer_ae_allha/results/test_metrics_global_hb_interp.json
  data/contrastive_g5_deepsad_transformer_ae_hb_diversity/results/test_metrics_global.json
  data/results/haus_b_adaptation/haus_ab_feature_distribution_summary.json
    (unaffected by the leak — all used global/no-recalibration eval modes)

Writes:
  data/results/haus_b_adaptation/haus_b_adaptation_comparison.xlsx

Run: python scripts/export_haus_b_adaptation_excel.py
"""
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
ALLHA_DIR = ROOT / 'data' / 'contrastive_g5_deepsad_transformer_ae_allha' / 'results'
DIVERSITY_DIR = ROOT / 'data' / 'contrastive_g5_deepsad_transformer_ae_hb_diversity' / 'results'
OUT_DIR = ROOT / 'data' / 'results' / 'haus_b_adaptation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_FILL = PatternFill('solid', fgColor='1F4E78')
HEADER_FONT = Font(color='FFFFFF', bold=True)
FAIL_FILL = PatternFill('solid', fgColor='F8D7DA')
SUCCESS_FILL = PatternFill('solid', fgColor='D4EDDA')
WARN_FILL = PatternFill('solid', fgColor='FFF3CD')
TITLE_FONT = Font(bold=True, size=13)
NOTE_FONT = Font(italic=True, color='555555', size=9)
THIN = Side(style='thin', color='CCCCCC')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _style_header(ws, row, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = BORDER


def _autofit(ws, n_cols, min_width=10, max_width=42):
    for c in range(1, n_cols + 1):
        col_letter = get_column_letter(c)
        max_len = min_width
        for cell in ws[col_letter]:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)) + 2)
        ws.column_dimensions[col_letter].width = min(max_len, max_width)


def load(path):
    with open(path) as f:
        return json.load(f)


def write_metrics_sheet(wb, title, data, fill, note=None, extra_cols=None):
    """extra_cols: list of (header, key_in_metric_dict) appended after the standard columns."""
    ws = wb.create_sheet(title[:31])
    row = 1
    ws.cell(row=row, column=1, value=title).font = TITLE_FONT
    row += 1
    if note:
        ws.cell(row=row, column=1, value=note).font = NOTE_FONT
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
        row += 1
    row += 1

    cols = ['Series', 'ROC-AUC', 'F1', 'Accuracy', 'N Correct', 'N Wrong',
            'TP', 'TN', 'FP', 'FN', 'N NoFlow', 'N Flow', 'Threshold']
    if extra_cols:
        cols = cols + [c[0] for c in extra_cols]
    header_row = row
    for c, name in enumerate(cols, start=1):
        ws.cell(row=header_row, column=c, value=name)
    _style_header(ws, header_row, len(cols))
    row += 1

    overall = data.pop('overall', None)
    for series, m in data.items():
        cm = m.get('confusion_matrix', {})
        vals = [
            series, round(m.get('roc_auc', 0), 4), round(m.get('f1', 0), 4),
            round(m.get('accuracy', 0), 4), m.get('n_correct'), m.get('n_wrong'),
            cm.get('tp'), cm.get('tn'), cm.get('fp'), cm.get('fn'),
            m.get('n_noflow'), m.get('n_flow'), round(m.get('threshold', 0), 2),
        ]
        if extra_cols:
            vals += [m.get(key) for _, key in extra_cols]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border = BORDER
            if c == 2:  # ROC-AUC column, colour by pass/fail
                cell.fill = SUCCESS_FILL if (v is not None and v >= 0.85) else FAIL_FILL
        row += 1

    if overall:
        row += 1
        ws.cell(row=row, column=1, value='OVERALL').font = Font(bold=True)
        ws.cell(row=row, column=2, value=round(overall.get('mean_roc_auc', 0), 4)).font = Font(bold=True)
        ws.cell(row=row, column=3, value=round(overall.get('mean_f1', 0), 4)).font = Font(bold=True)
        ws.cell(row=row, column=4, value=round(overall.get('mean_accuracy', 0), 4)).font = Font(bold=True)
        ws.cell(row=row, column=5, value=overall.get('total_n_correct')).font = Font(bold=True)
        ws.cell(row=row, column=6, value=overall.get('total_n_wrong')).font = Font(bold=True)
        cm = overall.get('confusion_matrix', {})
        ws.cell(row=row, column=7, value=cm.get('tp')).font = Font(bold=True)
        ws.cell(row=row, column=8, value=cm.get('tn')).font = Font(bold=True)
        ws.cell(row=row, column=9, value=cm.get('fp')).font = Font(bold=True)
        ws.cell(row=row, column=10, value=cm.get('fn')).font = Font(bold=True)
        if 'mean_specificity' in overall:
            ws.cell(row=row, column=len(cols), value=round(overall['mean_specificity'], 4)).font = Font(bold=True)
        for c in range(1, len(cols) + 1):
            ws.cell(row=row, column=c).fill = fill

    ws.freeze_panes = f'A{header_row + 1}'
    _autofit(ws, len(cols))


def write_mv_sheet(wb, title, mv_data, subtitle):
    ws = wb.create_sheet(title[:31])
    row = 1
    ws.cell(row=row, column=1, value=subtitle).font = TITLE_FONT
    row += 2

    cols = ['Series', 'File', 'True Label', 'Majority Prediction', 'Correct?', 'N NoFlow Patches', 'N Leakage Patches']
    header_row = row
    for c, name in enumerate(cols, start=1):
        ws.cell(row=header_row, column=c, value=name)
    _style_header(ws, header_row, len(cols))
    row += 1

    n_correct_total, n_total = 0, 0
    for series, files in mv_data.items():
        for fname, v in files.items():
            correct = v['true_label'] == v['majority_pred']
            n_total += 1
            n_correct_total += int(correct)
            vals = [series, fname, v['true_label'], v['majority_pred'],
                    'YES' if correct else 'NO', v.get('n_noflow'), v.get('n_leakage')]
            for c, val in enumerate(vals, start=1):
                cell = ws.cell(row=row, column=c, value=val)
                cell.border = BORDER
                if c == 5:
                    cell.fill = SUCCESS_FILL if correct else FAIL_FILL
            row += 1

    row += 1
    ws.cell(row=row, column=1, value='TOTAL').font = Font(bold=True)
    if n_total:
        ws.cell(row=row, column=2, value=f'{n_correct_total}/{n_total} files correct '
                                          f'({100 * n_correct_total / n_total:.1f}%)').font = Font(bold=True)

    ws.freeze_panes = f'A{header_row + 1}'
    _autofit(ws, len(cols))


def write_fair_fold_detail_sheet(wb, folds):
    ws = wb.create_sheet('Per-Site FAIR Fold Detail')
    row = 1
    ws.cell(row=row, column=1,
            value='Fair Per-Site Evaluation — Full Calibration/Test Audit Trail').font = TITLE_FONT
    row += 1
    ws.cell(row=row, column=1, value=(
        'Calibration = earliest NoFlow files (fit GMM/threshold only, NEVER scored). '
        'Test NoFlow + all Leak files = scored. Leak files can fall chronologically between '
        'NoFlow days in real data (e.g. hb-25.05.2026) — that is expected, not an error.'
    )).font = NOTE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 2

    cols = ['Series', 'Calibration Files', 'Test NoFlow Files', 'Leak Files',
            'N Calib Patches', 'N Test NoFlow Patches', 'N Leak Patches',
            'Calibration Date Range', 'Test NoFlow Date Range', 'GMM Mode',
            'Flagged Small Calibration?', 'Threshold', 'ROC-AUC', 'F1', 'Specificity', 'Skipped Reason']
    header_row = row
    for c, name in enumerate(cols, start=1):
        ws.cell(row=header_row, column=c, value=name)
    _style_header(ws, header_row, len(cols))
    row += 1

    for series, f in folds.items():
        skipped = 'skipped_reason' in f
        vals = [
            series,
            ', '.join(f.get('calibration_files', [])),
            ', '.join(f.get('test_noflow_files', [])),
            ', '.join(f.get('leak_files', [])),
            f.get('n_calibration_patches'), f.get('n_test_noflow_patches'), f.get('n_leak_patches'),
            ' to '.join(str(d) for d in f.get('calibration_date_range', [None, None])),
            ' to '.join(str(d) for d in f.get('test_noflow_date_range', [None, None])),
            f.get('gmm_mode', ''),
            'YES' if f.get('flagged_small_calibration') else '',
            round(f['threshold'], 3) if 'threshold' in f else '',
            round(f['roc_auc'], 4) if 'roc_auc' in f else '',
            round(f['f1'], 4) if 'f1' in f else '',
            round(f['specificity'], 4) if 'specificity' in f else '',
            f.get('skipped_reason', ''),
        ]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            if c == 13 and v != '':  # ROC-AUC
                cell.fill = SUCCESS_FILL if v >= 0.85 else FAIL_FILL
            if c == 11 and v == 'YES':
                cell.fill = WARN_FILL
        if skipped:
            for c in range(1, len(cols) + 1):
                ws.cell(row=row, column=c).fill = PatternFill('solid', fgColor='E9ECEF')
        row += 1

    ws.freeze_panes = f'A{header_row + 1}'
    _autofit(ws, len(cols), max_width=55)


def write_center_check_sheet(wb, cc):
    ws = wb.create_sheet('Center Check (Part 1)')
    row = 1
    ws.cell(row=row, column=1, value='Pooled Two-House DeepSAD Center Diagnostic').font = TITLE_FONT
    row += 2
    ws.cell(row=row, column=1, value='Center source:').font = Font(bold=True)
    ws.cell(row=row, column=2, value=cc['center_source'].split(' — ')[0])
    row += 1
    if ' — ' in cc['center_source']:
        ws.cell(row=row, column=1, value=cc['center_source'].split(' — ', 1)[1]).font = NOTE_FONT
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
        row += 1
    row += 1

    rows = [
        ('N Haus A NoFlow patches', cc['n_haus_a_noflow']),
        ('N Haus B NoFlow patches', cc['n_haus_b_noflow']),
        ('Distance: Haus A mean -> center (d_A)', round(cc['d_A_to_center'], 5)),
        ('Distance: Haus B mean -> center (d_B)', round(cc['d_B_to_center'], 5)),
        ('Spread (Haus A, own mean)', round(cc['spread_A'], 5)),
        ('Spread (Haus B, own mean)', round(cc['spread_B'], 5)),
        ('Ratio d_A / spread_A', round(cc['ratio_dA_over_spreadA'], 4)),
        ('Ratio d_B / spread_B', round(cc['ratio_dB_over_spreadB'], 4)),
        ('Flagged: two separate clouds?', 'YES' if cc['flag_two_separate_clouds'] else 'no'),
    ]
    for label, val in rows:
        ws.cell(row=row, column=1, value=label).font = Font(bold=True)
        cell = ws.cell(row=row, column=2, value=val)
        cell.border = BORDER
        row += 1

    row += 1
    ws.cell(row=row, column=1, value=cc['reading']).font = NOTE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 2
    ws.cell(row=row, column=1, value=(
        'Reading: Haus A sits tightly around the shared center (ratio 0.28 — well anchored). '
        'Haus B is looser (ratio ~1.07 — about one full spread-width from center) — clearly the '
        'less-anchored house, though not over the >2.0 flag threshold on its own.'
    )).font = NOTE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)

    _autofit(ws, 6)


def write_pc1_temperature_sheet(wb, pt):
    ws = wb.create_sheet('PC1 vs Temperature')
    row = 1
    ws.cell(row=row, column=1, value='Does Temperature Explain the Pooled Two-House Latent Drift?').font = TITLE_FONT
    row += 1
    ws.cell(row=row, column=1, value=(
        f"Per-series NoFlow centroids ({pt['n_series']} series: 17 Haus A + 2 Haus B), PCA fit on those "
        f"centroids, correlated against each series' mean {pt['temp_target']}. Same methodology as the "
        'established scripts/analyze_drift_temperature.py, applied to the pooled two-house DeepSAD '
        'Transformer checkpoint.'
    )).font = NOTE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 2

    ws.cell(row=row, column=1, value='PCA explained variance:').font = Font(bold=True)
    ws.cell(row=row, column=2, value=f"PC1={pt['pca_explained_variance_ratio'][0]:.1%}")
    ws.cell(row=row, column=3, value=f"PC2={pt['pca_explained_variance_ratio'][1]:.1%}")
    row += 2

    corr_cols = ['', 'Pearson r', 'Pearson p', 'Spearman ρ', 'Spearman p']
    header_row = row
    for c, name in enumerate(corr_cols, start=1):
        ws.cell(row=header_row, column=c, value=name)
    _style_header(ws, header_row, len(corr_cols))
    row += 1
    for label, prefix in [('PC1 vs ' + pt['temp_target'], 'pc1'), ('PC2 vs ' + pt['temp_target'], 'pc2')]:
        vals = [label, round(pt[f'{prefix}_vs_temp_pearson_r'], 4), round(pt[f'{prefix}_vs_temp_pearson_p'], 4),
                round(pt[f'{prefix}_vs_temp_spearman_r'], 4), round(pt[f'{prefix}_vs_temp_spearman_p'], 4)]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border = BORDER
            if c in (2, 4):  # correlation coefficients — flag |r| >= 0.5 as notable
                cell.fill = WARN_FILL if abs(v) >= 0.5 else SUCCESS_FILL
        row += 1
    row += 1

    ws.cell(row=row, column=1, value=(
        'Reading: PC1 shows a moderate correlation (Spearman significant at p=0.013) but the plot shows '
        'this is mostly driven by ONE outlier series (24.01.2026, coldest series, far highest PC1) — the '
        'rest of the 17 points look flat across the temperature range. Much weaker/messier than the earlier '
        'CNN encoder finding (PC2 vs temp_pipe, ρ=-0.90, clean relationship). PC2 shows no relationship at '
        'all here. hb-17.05 and ha-17.05 sit at nearly the same temperature (~19.05-19.09°C) but noticeably '
        'different PC1 (-0.033 vs +0.012) — house identity still matters beyond temperature alone.'
    )).font = NOTE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 3

    cols = ['Series', 'House', f"Mean {pt['temp_target']} [°C]", 'PC1', 'PC2', 'N NoFlow Patches']
    header_row = row
    for c, name in enumerate(cols, start=1):
        ws.cell(row=header_row, column=c, value=name)
    _style_header(ws, header_row, len(cols))
    row += 1
    for r in sorted(pt['per_series'], key=lambda x: x['temp']):
        vals = [r['series'], r['house'], round(r['temp'], 2), round(r['pc1'], 5), round(r['pc2'], 5),
                r['n_noflow_patches']]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border = BORDER
        row += 1

    ws.freeze_panes = f'A{header_row + 1}'
    _autofit(ws, len(cols))


def write_summary_sheet(wb, rows):
    ws = wb.create_sheet('Summary', 0)
    row = 1
    ws.cell(row=row, column=1, value='Haus B Adaptation — Summary (FAIR/corrected results are the headline)').font = TITLE_FONT
    row += 1
    ws.cell(row=row, column=1, value=(
        'The earlier "0.945 AUC per-site recalibration works" result used a leaky evaluation '
        '(_evaluate_on_test: fit GMM on a series\' NoFlow, then scored that SAME NoFlow again). '
        'Corrected 2026-07-21: mean fair AUC = 0.305 (worse than random on 2 of 3 scored series). '
        'The OPTIMISTIC_OLD row below is kept only for side-by-side comparison, not a valid result.'
    )).font = NOTE_FONT
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 2

    cols = ['Approach', 'Verdict', 'Mean ROC-AUC', 'Mean F1', 'Majority Vote', 'Notes']
    header_row = row
    for c, name in enumerate(cols, start=1):
        ws.cell(row=header_row, column=c, value=name)
    _style_header(ws, header_row, len(cols))
    row += 1

    fill_map = {'SUCCESS': SUCCESS_FILL, 'FAILED': FAIL_FILL, 'INVALID': WARN_FILL}
    for r in rows:
        fill = fill_map.get(r['verdict'], FAIL_FILL)
        vals = [r['approach'], r['verdict'], r['auc'], r['f1'], r['mv'], r['notes']]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=row, column=c, value=v)
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            if c == 2:
                cell.fill = fill
                cell.font = Font(bold=True)
        row += 1

    ws.freeze_panes = f'A{header_row + 1}'
    _autofit(ws, len(cols), max_width=60)


def main():
    wb = Workbook()
    wb.remove(wb.active)

    def _mean(d, key):
        return round(d.get('overall', {}).get(key, 0), 4)

    # --- Part 1: center check ---
    cc_path = OUT_DIR / 'deepsad_center_check.json'
    if cc_path.exists():
        write_center_check_sheet(wb, load(cc_path))

    pt_path = OUT_DIR / 'center_check_pc1_temperature.json'
    if pt_path.exists():
        write_pc1_temperature_sheet(wb, load(pt_path))

    # --- Part 2: FAIR (corrected, current headline) ---
    fair_path = ALLHA_DIR / 'test_metrics.json'
    fair_full = load(fair_path) if fair_path.exists() else None
    if fair_full:
        write_metrics_sheet(
            wb, 'Per-Site FAIR (corrected)', dict(fair_full), FAIL_FILL,
            note='Calibration = earliest NoFlow files, scored ONLY on later disjoint NoFlow + all Leak files. '
                 'This replaces the leaky per-site result below as the headline (2026-07-21).',
            extra_cols=[('Specificity', 'specificity'), ('GMM Mode', 'gmm_mode')],
        )
    folds_path = ALLHA_DIR / 'test_metrics_fair_folds.json'
    if folds_path.exists():
        write_fair_fold_detail_sheet(wb, load(folds_path))
    mv_fair_path = ALLHA_DIR / 'majority_vote_results.json'
    mv_fair = load(mv_fair_path) if mv_fair_path.exists() else None
    if mv_fair:
        write_mv_sheet(wb, 'Per-Site FAIR Majority Vote', mv_fair,
                        'Fair Per-Site Evaluation — Majority Vote (per file, calibration files excluded)')

    # --- Part 2: OPTIMISTIC_OLD (leaky, kept only for comparison) ---
    opt_path = ALLHA_DIR / 'test_metrics_OPTIMISTIC_OLD.json'
    opt_full = load(opt_path) if opt_path.exists() else None
    if opt_full:
        write_metrics_sheet(
            wb, 'Per-Site OPTIMISTIC_OLD (leaky)', dict(opt_full), WARN_FILL,
            note='INVALID — kept only for comparison. Fit the GMM on a series\' NoFlow and then scored that '
                 'SAME NoFlow again, inflating specificity/AUC. Do not cite these numbers.',
        )

    diversity_path = DIVERSITY_DIR / 'test_metrics_global.json'
    if diversity_path.exists():
        d = load(diversity_path)
        write_metrics_sheet(
            wb, 'Global training-diversity (FAILED)', dict(d), FAIL_FILL,
            note='Encoder trained with 2 Haus B leak series folded into training (same recipe that fixed '
                 'Haus A seasonal drift); scored with global GMM, no per-site refit. Uses global eval mode, '
                 'unaffected by the per-site leak.',
        )

    extrap_path = ALLHA_DIR / 'test_metrics_global_hb_extrap.json'
    if extrap_path.exists():
        d = load(extrap_path)
        write_metrics_sheet(
            wb, 'HB-only Extrapolation (FAILED)', dict(d), FAIL_FILL,
            note='GMM fit ONLY on early Haus B NoFlow-only series (hb-26.04, hb-4.05, hb-10.05, no Haus A at all), '
                 'scored on the 4 later Haus B leak series, no further refit. Uses global eval mode, unaffected '
                 'by the per-site leak.',
        )

    interp_path = ALLHA_DIR / 'test_metrics_global_hb_interp.json'
    if interp_path.exists():
        d = load(interp_path)
        write_metrics_sheet(
            wb, 'HB-only Interpolation (FAILED)', dict(d), FAIL_FILL,
            note='Same as HB-only Extrapolation, plus hb-7.06.2026 added to the fit pool. Uses global eval '
                 'mode, unaffected by the per-site leak.',
        )

    feat_path = OUT_DIR / 'haus_ab_feature_distribution_summary.json'
    if feat_path.exists():
        fd = load(feat_path)
        ws = wb.create_sheet('Feature Distribution Check')
        row = 1
        ws.cell(row=row, column=1,
                value='Raw G5 Feature Distribution — Haus A vs. Haus B NoFlow (pre-encoder)').font = TITLE_FONT
        row += 2
        sm = fd['scaler_mismatch']
        ws.cell(row=row, column=1, value='Scaler fit on Haus A NoFlow only, then applied to both houses:').font = Font(bold=True)
        row += 1
        for label, key in [
            ('Haus A scaled mean (should be ~0)', 'haus_a_scaled_mean_overall'),
            ('Haus A scaled std (should be ~1)', 'haus_a_scaled_std_overall'),
            ('Haus B scaled mean', 'haus_b_scaled_mean_overall'),
            ('Haus B scaled std', 'haus_b_scaled_std_overall'),
        ]:
            ws.cell(row=row, column=1, value=label)
            ws.cell(row=row, column=2, value=round(sm[key], 4))
            row += 1
        row += 1
        ws.cell(row=row, column=1, value=sm['note']).font = NOTE_FONT
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
        row += 3

        cols = ['Feature', 'Haus A Mean', 'Haus A Std', 'Haus B Mean', 'Haus B Std', 'Gap (in Haus-A std units)']
        header_row = row
        for c, name in enumerate(cols, start=1):
            ws.cell(row=header_row, column=c, value=name)
        _style_header(ws, header_row, len(cols))
        row += 1
        for r in sorted(fd['per_feature'], key=lambda x: -abs(x['mean_gap_in_haus_a_std'])):
            vals = [r['feature'], round(r['haus_a_mean'], 6), round(r['haus_a_std'], 6),
                    round(r['haus_b_mean'], 6), round(r['haus_b_std'], 6), round(r['mean_gap_in_haus_a_std'], 3)]
            for c, v in enumerate(vals, start=1):
                cell = ws.cell(row=row, column=c, value=v)
                cell.border = BORDER
                if c == 6:
                    cell.fill = FAIL_FILL if abs(v) > 5 else SUCCESS_FILL
            row += 1
        ws.freeze_panes = f'A{header_row + 1}'
        _autofit(ws, len(cols))

    # --- Summary ---
    summary_rows = []

    if fair_full:
        mv_correct = sum(1 for files in (mv_fair or {}).values()
                          for v in files.values() if v['true_label'] == v['majority_pred'])
        mv_total = sum(len(files) for files in (mv_fair or {}).values())
        summary_rows.append({
            'approach': 'Per-site FAIR (corrected)\n(calibration/test NoFlow disjoint, chronological)',
            'verdict': 'FAILED',
            'auc': _mean(fair_full, 'mean_roc_auc'),
            'f1': _mean(fair_full, 'mean_f1'),
            'mv': f'{mv_correct}/{mv_total} files ({100 * mv_correct / mv_total:.0f}%)' if mv_total else 'n/a',
            'notes': 'Current headline (2026-07-21). hb-14.06 & hb-25.05 collapse to near-total inversion '
                     '(specificity=0.0 — every test NoFlow file misclassified). hb-31.05 skipped (only 1 '
                     'usable NoFlow file, can\'t split fairly). Shows Haus B\'s NoFlow baseline can go stale '
                     'in 1-2 DAYS, not just between separate visits.',
        })
    if opt_full:
        summary_rows.append({
            'approach': 'Per-site OPTIMISTIC_OLD (leaky)\n(fit AND scored on the same NoFlow)',
            'verdict': 'INVALID',
            'auc': _mean(opt_full, 'mean_roc_auc'),
            'f1': _mean(opt_full, 'mean_f1'),
            'mv': 'n/a (not re-collected for this run)',
            'notes': 'Kept ONLY for comparison against the fair result above. Do not cite. Previously reported '
                     'as the Haus B "solution" — that conclusion is now known to be a leakage artefact.',
        })
    if diversity_path.exists():
        d = load(diversity_path)
        summary_rows.append({
            'approach': 'Global GMM, training-diversity\n(2 Haus B leak series added to encoder training)',
            'verdict': 'FAILED',
            'auc': _mean(d, 'mean_roc_auc'),
            'f1': _mean(d, 'mean_f1'),
            'mv': 'n/a (not saved for this mode)',
            'notes': 'Same recipe that fixed Haus A seasonal drift (Eval A->B); does not transfer to Haus B.',
        })
    if extrap_path.exists():
        d = load(extrap_path)
        summary_rows.append({
            'approach': 'HB-only extrapolation\n(GMM fit on 3 early Haus B NoFlow series only)',
            'verdict': 'FAILED',
            'auc': _mean(d, 'mean_roc_auc'),
            'f1': _mean(d, 'mean_f1'),
            'mv': 'n/a (not saved for this mode)',
            'notes': 'Works for the nearest-in-time test series, collapses for series weeks later.',
        })
    if interp_path.exists():
        d = load(interp_path)
        summary_rows.append({
            'approach': 'HB-only interpolation\n(adds hb-7.06 to the fit pool)',
            'verdict': 'FAILED',
            'auc': _mean(d, 'mean_roc_auc'),
            'f1': _mean(d, 'mean_f1'),
            'mv': 'n/a (not saved for this mode)',
            'notes': 'Same pattern as extrapolation.',
        })
    summary_rows.append({
        'approach': 'Global GMM, no adaptation\n(original Track 1, 2026-06-16)',
        'verdict': 'FAILED',
        'auc': 'n/a (raw file not retained locally)',
        'f1': 'n/a',
        'mv': '0/21 files (all flagged LEAKAGE incl. NoFlow)',
        'notes': 'Complete failure — all 21 files across 3 series flagged LEAKAGE with p_mean=1.000.',
    })

    write_summary_sheet(wb, summary_rows)

    out_path = OUT_DIR / 'haus_b_adaptation_comparison.xlsx'
    wb.save(out_path)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
