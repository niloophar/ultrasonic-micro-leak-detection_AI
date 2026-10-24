#!/usr/bin/env python
"""
Thesis figure — end-to-end method pipeline for micro-leak detection.

Two-row layout (4 + 3 stages) keeps the figure compact for a thesis
page width instead of one long horizontal strip.

Outputs:
  figures/method_pipeline.pdf
  figures/method_pipeline.png

Run: python scripts/plot_thesis_method_pipeline.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ─── colours ──────────────────────────────────────────────────────────────────
C_EDGE   = '#202020'
C_TEXT   = '#1a1a1a'
C_SUB    = '#6e6e6e'
C_FILL   = '#f4f4f5'
C_BRANCH = '#8a8a8a'

LW_BOX = 1.2
LW_ARR = 1.2

TITLE_FS = 10.3
SUB_FS = 7.9

# ─── stages, split into two reading rows ─────────────────────────────────────
ROW1 = [
    ('Zero-crossing data',   'upstream/downstream ToF hits'),
    ('Patch construction',   'block aggregation'),
    ('Normalized intervals', 'relative ZC spacing'),
    ('Encoder',              'contrastive training'),
]
ROW2 = [
    ('Latent space',  'embedding z'),
    ('GMM scorer',    'NoFlow density'),
    ('Anomaly score', 'low likelihood = leakage'),
]

BOX_W, BOX_H = 2.3, 1.4
X_GAP = 0.55
X0 = 0.5

xs1 = [X0 + i * (BOX_W + X_GAP) for i in range(len(ROW1))]
xs2 = xs1[:len(ROW2)]  # row 2 left-aligned under the first 3 columns of row 1

FIG_W = xs1[-1] + BOX_W + X0

BR_W, BR_H = 2.6, 1.2
Y_BRANCH = 0.3
GAP_BRANCH_ROW2 = 0.75
Y_ROW2 = Y_BRANCH + BR_H + GAP_BRANCH_ROW2
GAP_ROW2_ROW1 = 0.7
Y_ROW1 = Y_ROW2 + BOX_H + GAP_ROW2_ROW1
FIG_H = Y_ROW1 + BOX_H + 0.3

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')


def draw_box(x, y, w, h, title, subtitle, dashed=False, fill=C_FILL,
             title_fs=TITLE_FS, sub_fs=SUB_FS):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.11',
        linewidth=LW_BOX,
        edgecolor=C_EDGE,
        facecolor=fill,
        linestyle='--' if dashed else '-',
        zorder=3,
    )
    ax.add_patch(box)
    cx = x + w / 2
    ax.text(cx, y + h * 0.66, title, ha='center', va='center',
            fontsize=title_fs, color=C_TEXT, fontweight='bold', zorder=4)
    ax.text(cx, y + h * 0.30, subtitle, ha='center', va='center',
            fontsize=sub_fs, color=C_SUB, style='italic', zorder=4)


def draw_arrow(x_from, x_to, y, color=C_EDGE, dashed=False, mscale=13):
    arr = FancyArrowPatch(
        (x_from, y), (x_to, y),
        arrowstyle='-|>', mutation_scale=mscale,
        linewidth=LW_ARR, color=color,
        linestyle='--' if dashed else '-',
        zorder=2,
    )
    ax.add_patch(arr)


# ─── row 1 ───────────────────────────────────────────────────────────────────
for i, (title, subtitle) in enumerate(ROW1):
    draw_box(xs1[i], Y_ROW1, BOX_W, BOX_H, title, subtitle)
for i in range(len(ROW1) - 1):
    draw_arrow(xs1[i] + BOX_W + 0.03, xs1[i + 1] - 0.03, Y_ROW1 + BOX_H / 2)

# ─── row 2 ───────────────────────────────────────────────────────────────────
for i, (title, subtitle) in enumerate(ROW2):
    draw_box(xs2[i], Y_ROW2, BOX_W, BOX_H, title, subtitle)
for i in range(len(ROW2) - 1):
    draw_arrow(xs2[i] + BOX_W + 0.03, xs2[i + 1] - 0.03, Y_ROW2 + BOX_H / 2)

# ─── row-wrap connector: Encoder (row1, last) → Latent space (row2, first) ───
x_encoder = xs1[-1] + BOX_W / 2
x_latent = xs2[0] + BOX_W / 2
gutter_y = (Y_ROW2 + BOX_H + Y_ROW1) / 2

ax.add_line(Line2D([x_encoder, x_encoder], [Y_ROW1, gutter_y],
                    color=C_EDGE, lw=LW_ARR, zorder=2))
ax.add_line(Line2D([x_encoder, x_latent], [gutter_y, gutter_y],
                    color=C_EDGE, lw=LW_ARR, zorder=2))
arr = FancyArrowPatch(
    (x_latent, gutter_y), (x_latent, Y_ROW2 + BOX_H + 0.03),
    arrowstyle='-|>', mutation_scale=13, linewidth=LW_ARR, color=C_EDGE, zorder=2,
)
ax.add_patch(arr)

# ─── optional lower branch: site-wise recalibration ──────────────────────────
gmm_x = xs2[1]
gmm_cx = gmm_x + BOX_W / 2
br_x = gmm_cx - BR_W / 2

draw_box(br_x, Y_BRANCH, BR_W, BR_H,
         'Site-wise recalibration', 'initial NoFlow window',
         dashed=True, fill='white', title_fs=9.4, sub_fs=SUB_FS)

arr = FancyArrowPatch(
    (gmm_cx, Y_BRANCH + BR_H), (gmm_cx, Y_ROW2),
    arrowstyle='-|>', mutation_scale=12,
    linewidth=LW_ARR, color=C_BRANCH, linestyle='--', zorder=2,
)
ax.add_patch(arr)

ax.text(gmm_cx + 0.18, (Y_BRANCH + BR_H + Y_ROW2) / 2,
        'scorer only\n(deployment)',
        ha='left', va='center', fontsize=7.3, color=C_BRANCH, style='italic')

# ─── save ─────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 300)]:
    p = out_dir / f'method_pipeline.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
