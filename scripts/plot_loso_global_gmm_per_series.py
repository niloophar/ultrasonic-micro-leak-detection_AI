#!/usr/bin/env python
"""
Final thesis figure for the House A LOSO interpolation section: per-series
ROC-AUC (+ F1) under the DeepSAD-Transformer LOSO global-GMM protocol
(one GMM per fold, fitted on the training pool's NoFlow only, no per-site
recalibration on the held-out series).

Does NOT rerun or modify any trained model, GMM fit, or stored metric value
— reads only the existing loso_results.json.

Reads:
  data/results/interpolation_global_gmm/results/loso_results.json

Writes:
  figures/loso_global_gmm_per_series.pdf
  figures/loso_global_gmm_per_series.png

Run: python scripts/plot_loso_global_gmm_per_series.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
LOSO_PATH = ROOT / 'data' / 'results' / 'interpolation_global_gmm' / 'results' / 'loso_results.json'
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Chronological order of the 17-series loso_series pool (all Haus A, excl.
# 30.03 — estimated labels, never trusted). ha-9.05 (no data on disk) and
# ha-24.05 (NoFlow-only, AUC undefined) are excluded below at load time,
# matching every other AUC-mean reported for this experiment in CLAUDE.md.
CHRONO_ORDER = [
    '10.01.2026', '24.01.2026', '7.02.2026', '21.02.2026', '7.03.2026',
    '15.03.2026', '5.04.2026', '11.04.2026', '18.04.2026', 'ha-25.04.2026',
    'ha-2.05.2026', 'ha-9.05.2026', 'ha-17.05.2026', 'ha-24.05.2026',
    'ha-2.06.2026', 'ha-6.06.2026', 'ha-20.06.2026',
]
SHORT_LABEL = {s: s.replace('.2026', '') for s in CHRONO_ORDER}
OUTLIER_SERIES = {'ha-2.06.2026', 'ha-20.06.2026'}

# ─── calm, thesis-consistent categorical pair (same validated pair used in
# the CNN-vs-Transformer figure: node validate_palette.js "#2a78d6,#eb6834"
# --mode light -> all checks pass) ──────────────────────────────────────────
C_AUC = '#2a78d6'    # blue   — ROC-AUC (primary metric)
C_F1 = '#eb6834'     # orange — F1 (secondary metric)
C_DARK = '#1a1a1a'
C_GRAY = '#777777'
C_LGRAY = '#cccccc'
C_OUTLIER_BG = '#fbe4e0'   # very light warm tint — outlier highlight band
C_OUTLIER_TXT = '#b23a1f'  # muted dark red-orange — outlier label color

with open(LOSO_PATH) as f:
    loso = json.load(f)
loso.pop('_summary', None)

series = [s for s in CHRONO_ORDER if loso.get(s)]
aucs = [loso[s]['roc_auc'] for s in series]
f1s = [loso[s]['f1'] for s in series]
skipped = [s for s in CHRONO_ORDER if not loso.get(s)]
print(f'{len(series)} AUC-computable series plotted (chronological).')
print('Skipped (no data / NoFlow-only, not AUC-computable):', skipped)

x = list(range(len(series)))
labels = [SHORT_LABEL[s] for s in series]

fig, ax = plt.subplots(figsize=(11, 5.2))

# ─── outlier highlight bands (drawn first, behind everything) ─────────────
for i, s in enumerate(series):
    if s in OUTLIER_SERIES:
        ax.axvspan(i - 0.5, i + 0.5, color=C_OUTLIER_BG, zorder=0)

# ─── ROC-AUC bars (main metric) ────────────────────────────────────────────
bars = ax.bar(x, aucs, width=0.6, color=C_AUC, edgecolor='white', linewidth=0.6, zorder=3,
              label='ROC-AUC')
for xi, v in zip(x, aucs):
    ax.text(xi, min(v + 0.03, 0.99), f'{v:.2f}', ha='center', va='bottom',
            fontsize=7.5, color=C_DARK, zorder=4)

# ─── F1 secondary line + markers ──────────────────────────────────────────
ax.plot(x, f1s, color=C_F1, marker='o', markersize=5, markerfacecolor=C_F1,
        markeredgecolor='white', markeredgewidth=0.8, linewidth=1.6, zorder=5, label='F1')

# ─── outlier annotation ─────────────────────────────────────────────────
# Placed near the top of the highlighted band rather than near the bars'
# own base: both flagged series have short/medium bars, but a fixed
# low-y position would still sit inside or right against the taller of
# the two (ha-20.06, AUC 0.28) once its "score\ninversion" text height is
# accounted for. Anchoring near the top instead guarantees clearance from
# the bar, its value label, and the F1 marker for both series regardless
# of their individual bar heights.
for i, s in enumerate(series):
    if s in OUTLIER_SERIES:
        ax.text(i, 0.95, 'score\ninversion', ha='center', va='top',
                fontsize=7.3, color=C_OUTLIER_TXT, fontweight='bold', linespacing=1.2, zorder=4)

# ─── axes styling ───────────────────────────────────────────────────────
ax.set_ylim(0.0, 1.0)
ax.set_xlim(-0.6, len(series) - 0.4)
ax.set_xticks(x)
xticklabels = ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
for i, s in enumerate(series):
    if s in OUTLIER_SERIES:
        xticklabels[i].set_color(C_OUTLIER_TXT)
        xticklabels[i].set_fontweight('bold')
ax.set_ylabel('Score', fontsize=11, color=C_DARK)
ax.set_xlabel('Held-out series (chronological)', fontsize=11, color=C_DARK)
ax.tick_params(axis='y', colors=C_DARK, labelsize=9.5)
ax.tick_params(axis='x', colors=C_DARK)
ax.grid(axis='y', color=C_LGRAY, lw=0.5, alpha=0.5, zorder=0)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color(C_LGRAY)
ax.spines['bottom'].set_color(C_LGRAY)

legend_handles = [
    Patch(facecolor=C_AUC, edgecolor='white', label='ROC-AUC'),
    Line2D([0], [0], color=C_F1, marker='o', markersize=5, markerfacecolor=C_F1,
           markeredgecolor='white', linewidth=1.6, label='F1'),
    Patch(facecolor=C_OUTLIER_BG, edgecolor='none', label='Score inversion'),
]
ax.legend(handles=legend_handles, loc='lower left', frameon=False, fontsize=9)

ax.set_title('House A LOSO — ROC-AUC and F1 by Held-Out Series', fontsize=13,
             fontweight='bold', color=C_DARK, pad=12)
fig.text(
    0.5, -0.01,
    'DeepSAD Transformer, 17-series loso_series pool (all Haus A excl. 30.03); global GMM, no per-site '
    'recalibration. ha-9.05 (no data) and ha-24.05 (NoFlow-only) omitted — AUC undefined for both.',
    ha='center', va='top', fontsize=7.8, color=C_GRAY, linespacing=1.4, transform=fig.transFigure,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
# Saved under both this script's own name AND `loso_hausa.*` -- the thesis
# (Figure 4.2, \includegraphics{loso_hausa.png}) embeds the latter. The
# on-disk file it used to resolve to was actually named "loso_hausa .png"
# (stray space before the extension), which \includegraphics{loso_hausa.png}
# would never match -- fixed here by writing the correctly-named file.
for ext in ('png', 'pdf'):
    kw = dict(format=ext, bbox_inches='tight')
    if ext == 'png':
        kw['dpi'] = 300
    for stem in ('loso_global_gmm_per_series', 'loso_hausa'):
        out_path = FIG_DIR / f'{stem}.{ext}'
        fig.savefig(out_path, **kw)
        print(f'Saved -> {out_path}')
