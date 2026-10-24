#!/usr/bin/env python
"""
Thesis figure -- protocol diagram contrasting fixed seasonal interpolation
vs. extrapolation splits.

Purely schematic: block width/spacing carries no quantitative meaning
(no durations, no metric values). Shares its visual vocabulary (palette,
panel style, legend layout) with the other thesis concept figures.

Two stacked panels:
  (a) Seasonal interpolation: training series -> held-out test series ->
      training series. The held-out series sits temporally INSIDE the
      training envelope, but is fully excluded from training.
  (b) Seasonal extrapolation: training series -> training series ->
      held-out future series. No training data exist after the held-out
      period; the held-out series may exceed the warmest/latest training
      condition.

Splitting always occurs at the whole-recording-series level, never at
the patch level -- this is depicted by drawing each series as one solid
block, never mixing train/test fill within a single block.

Outputs:
  figures/fixed_interpolation_extrapolation_protocol.pdf
  figures/fixed_interpolation_extrapolation_protocol.png (600 dpi)

Run: python scripts/plot_thesis_interpolation_extrapolation_protocol.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ─── palette (shared with the other concept figures) ────────────────────────
C_TEXT = '#232323'
C_SUB = '#5c5c5c'
C_MUTED = '#8f8f8f'

C_TRAIN_FILL = '#cfe0f2'
C_TRAIN_EDGE = '#5b8fc2'
C_TEST_FILL = '#f6e6c9'
C_TEST_EDGE = '#b8860b'

C_PANEL_EDGE = '#9a9a9a'
C_PANEL_FILL = '#fbfbfa'
C_BOUNDARY = '#a1522f'

LW_PANEL = 1.2
TITLE_FS = 15.0
LABEL_FS = 10.5
SMALL_FS = 9.2
TINY_FS = 8.3

BLOCK_W, BLOCK_H = 1.55, 1.0
GAP = 0.32

# ─── figure setup ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 14.5, 7.9
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')


# ─── helpers ─────────────────────────────────────────────────────────────────
def draw_panel_box(x, y, w, h):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.20',
        linewidth=LW_PANEL, edgecolor=C_PANEL_EDGE, facecolor=C_PANEL_FILL,
        linestyle=(0, (5, 3)), zorder=0,
    )
    ax.add_patch(box)


def draw_block(x, y, kind='train'):
    """One block = one complete recording series. `kind` in {'train','test'}."""
    if kind == 'train':
        fill, edge, hatch, ls = C_TRAIN_FILL, C_TRAIN_EDGE, None, '-'
    else:
        fill, edge, hatch, ls = C_TEST_FILL, C_TEST_EDGE, '////', (0, (4, 2))
    box = FancyBboxPatch(
        (x, y), BLOCK_W, BLOCK_H, boxstyle='round,pad=0.0,rounding_size=0.06',
        linewidth=1.5, edgecolor=edge, facecolor=fill, linestyle=ls,
        hatch=hatch, zorder=3,
    )
    ax.add_patch(box)


def draw_time_arrow(x0, x1, y):
    arr = FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=11,
                           linewidth=1.1, color=C_MUTED, zorder=2, shrinkA=0, shrinkB=0)
    ax.add_patch(arr)
    ax.text(x1 + 0.10, y, 'time', ha='left', va='center', fontsize=TINY_FS,
             color=C_MUTED, style='italic')


def draw_envelope(x0, x1, y):
    tick_h = 0.10
    ax.add_line(Line2D([x0, x1], [y, y], color=C_SUB, lw=1.1, zorder=2))
    ax.add_line(Line2D([x0, x0], [y - tick_h, y + tick_h], color=C_SUB, lw=1.1, zorder=2))
    ax.add_line(Line2D([x1, x1], [y - tick_h, y + tick_h], color=C_SUB, lw=1.1, zorder=2))
    ax.text((x0 + x1) / 2, y + 0.16, 'training temporal envelope', ha='center', va='bottom',
             fontsize=SMALL_FS, color=C_SUB, style='italic')


def legend_swatch(x, y, kind, label):
    if kind == 'train':
        fill, edge, hatch, ls = C_TRAIN_FILL, C_TRAIN_EDGE, None, '-'
    else:
        fill, edge, hatch, ls = C_TEST_FILL, C_TEST_EDGE, '////', (0, (4, 2))
    box = FancyBboxPatch((x, y - 0.14), 0.30, 0.28, boxstyle='round,pad=0.0,rounding_size=0.03',
                          linewidth=1.3, edgecolor=edge, facecolor=fill, linestyle=ls,
                          hatch=hatch, zorder=6)
    ax.add_patch(box)
    ax.text(x + 0.42, y, label, ha='left', va='center', fontsize=SMALL_FS, color=C_TEXT)


# ═════════════════════════════════════════════════════════════════════════
# PANEL (a) — SEASONAL INTERPOLATION
# ═════════════════════════════════════════════════════════════════════════
PA_X, PA_Y, PA_W, PA_H = 0.4, 4.40, 13.7, 3.15
draw_panel_box(PA_X, PA_Y, PA_W, PA_H)
ax.text(PA_X + PA_W / 2, 7.68, '(a) Seasonal interpolation',
         ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

# Explicit bottom-up vertical budget within the panel box (PA_Y .. PA_Y+PA_H)
# so every element's position is checked to fit, rather than guessed offsets.
y_time_a = PA_Y + 0.33
y_annot_a = y_time_a + 0.32
by_a = y_annot_a + 0.30                      # block bottom edge
y_envelope_a = by_a + BLOCK_H + 0.22          # envelope tick line
y_envelope_label_a = y_envelope_a + 0.16      # envelope caption
# top used ≈ y_envelope_label_a + 0.22 text height; must stay under PA_Y+PA_H

seq_a = ['train', 'train', 'train', 'test', 'train', 'train']
n_a = len(seq_a)
total_w_a = n_a * BLOCK_W + (n_a - 1) * GAP
x0_a = PA_X + (PA_W - total_w_a) / 2

xs_a = []
for i, kind in enumerate(seq_a):
    x = x0_a + i * (BLOCK_W + GAP)
    xs_a.append(x)
    draw_block(x, by_a, kind)

draw_envelope(xs_a[0], xs_a[-1] + BLOCK_W, y_envelope_a)
draw_time_arrow(xs_a[0], xs_a[-1] + BLOCK_W, y_time_a)

test_x_a = xs_a[3]
ax.annotate('excluded from training', xy=(test_x_a + BLOCK_W / 2, by_a),
             xytext=(test_x_a + BLOCK_W / 2, y_annot_a),
             ha='center', va='bottom', fontsize=TINY_FS, color=C_TEST_EDGE, fontweight='bold',
             arrowprops=dict(arrowstyle='-', color=C_TEST_EDGE, lw=1.0))

legend_swatch(PA_X + 0.35, PA_Y + PA_H - 0.30, 'train', 'training series')
legend_swatch(PA_X + 0.35, PA_Y + PA_H - 0.66, 'test', 'held-out test series')

# ═════════════════════════════════════════════════════════════════════════
# PANEL (b) — SEASONAL EXTRAPOLATION
# ═════════════════════════════════════════════════════════════════════════
PB_X, PB_Y, PB_W, PB_H = 0.4, 0.55, 13.7, 3.15
draw_panel_box(PB_X, PB_Y, PB_W, PB_H)
ax.text(PB_X + PB_W / 2, 3.80, '(b) Seasonal extrapolation',
         ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

# Same explicit bottom-up budget as panel (a), but the held-out annotation
# needs 2 short lines here, so it gets a little more room.
y_time_b = PB_Y + 0.33
y_annot_b = y_time_b + 0.32
by_b = y_annot_b + 0.46                      # block bottom edge
y_envelope_b = by_b + BLOCK_H + 0.22
y_envelope_label_b = y_envelope_b + 0.16

seq_b = ['train', 'train', 'train', 'train', 'train']
n_b = len(seq_b)
FUTURE_GAP = 1.05
total_w_b = n_b * BLOCK_W + (n_b - 1) * GAP + FUTURE_GAP + BLOCK_W
x0_b = PB_X + (PB_W - total_w_b) / 2

xs_b = []
for i, kind in enumerate(seq_b):
    x = x0_b + i * (BLOCK_W + GAP)
    xs_b.append(x)
    draw_block(x, by_b, kind)

last_train_end = xs_b[-1] + BLOCK_W
test_x_b = last_train_end + FUTURE_GAP
draw_block(test_x_b, by_b, 'test')

# latest-training-boundary dashed line, label placed level with the
# envelope caption (same row, opposite side -- verified clear of it in x)
boundary_x = last_train_end + 0.12
ax.add_line(Line2D([boundary_x, boundary_x], [by_b - 0.15, y_envelope_label_b + 0.05],
                     color=C_BOUNDARY, lw=1.3, linestyle=(0, (4, 2)), zorder=4))
ax.text(boundary_x, y_envelope_label_b + 0.14, 'latest training boundary', ha='center', va='bottom',
         fontsize=TINY_FS, color=C_BOUNDARY)

draw_envelope(xs_b[0], last_train_end, y_envelope_b)
draw_time_arrow(xs_b[0], test_x_b + BLOCK_W, y_time_b)

ax.annotate('excluded — may exceed\ntraining conditions',
             xy=(test_x_b + BLOCK_W / 2, by_b), xytext=(test_x_b + BLOCK_W / 2, y_annot_b),
             ha='center', va='bottom', fontsize=TINY_FS, color=C_TEST_EDGE, fontweight='bold',
             linespacing=1.3,
             arrowprops=dict(arrowstyle='-', color=C_TEST_EDGE, lw=1.0))

# ─── save ───────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 600)]:
    p = out_dir / f'fixed_interpolation_extrapolation_protocol.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
