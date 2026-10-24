#!/usr/bin/env python
"""
Thesis figure — "Segment Extraction and Patch Generation": how one
equilibrium segment becomes a fixed-length encoder input patch.
Methodological diagram, not a data plot.

Every parameter shown is read from the actual pipeline rather than assumed:
  - configs/preparation/data_preparation.json: downsample_factor=16,
    patch_width=64, patch_overlap=0.75 (train only).
  - src/preparation/data_preparation.py::_prepare_patches: overlap is
    forced to 0.0 for any segment not in the train split (val/test always
    non-overlapping) — see the "Overlap only for training segments" branch.
  - src/preparation/data_preparation.py::downsample_segment: block
    aggregation is a plain per-block mean (stats=["mean"]).
  - configs/contrastive/encoder.json: selected_features = the 18 plain
    (non "__std") G5 columns (norm_interval_up/down_0..8), cnn_channel_mode
    = "dual" -> src/common/utils.py::reshape_to_channels with n_stats=1
    gives (N, 2, 9, W). With W=64 this is the (2, 9, 64) tensor shown here.

Run: python scripts/plot_thesis_patch_construction.py

Outputs:
    figures/feature_patch_construction.{pdf,png}
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# =============================================================================
# STYLE (Okabe-Ito colour-blind-safe palette; consistent with
# plot_thesis_feature_group_construction.py — same figure family)
# =============================================================================

C_EDGE = '#202020'
C_TEXT = '#1a1a1a'
C_SUB = '#6e6e6e'
C_FILL = '#f4f4f5'

C_UP = '#0072B2'      # blue — upstream
C_DOWN = '#D55E00'    # vermillion — downstream
C_AGG = '#CC79A7'     # reddish purple — temporal aggregation
C_TRAIN = '#E69F00'   # orange — training (overlapping) windows
C_EVAL = '#009E73'    # bluish green — validation/test (non-overlapping) windows

LW_BOX = 1.2
LW_ARR = 1.15

TITLE_FS = 10.0
SUB_FS = 8.0
MATH_FS = 9.5
SECTION_FS = 11.5
NOTE_FS = 7.3

plt.rcParams.update({
    'font.size': 9,
    'text.color': C_TEXT,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# =============================================================================
# LAYOUT — positions are computed at runtime via a running x-cursor (never
# hand-picked absolute coordinates), so stage widths can never silently
# overlap the next stage.
# =============================================================================

FIG_H = 7.6
Y_MID = 4.0
STAGE_GAP = 0.65

fig, ax = plt.subplots(figsize=(24, FIG_H))  # x-limits fixed up after layout
ax.axis('off')
ax.set_facecolor('white')


# =============================================================================
# HELPERS
# =============================================================================

def section_header(cx, text):
    ax.text(cx, FIG_H - 0.3, text, ha='center', va='top',
             fontsize=SECTION_FS, fontweight='bold', color=C_TEXT)


def divider(x):
    ax.plot([x, x], [0.3, FIG_H - 0.65], color='#d9d9d9', lw=0.9, zorder=0)


def straight_arrow(p0, p1, color=C_EDGE, lw=LW_ARR, mscale=12, ls='-', zorder=2):
    arr = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=mscale,
                            linewidth=lw, color=color, linestyle=ls, zorder=zorder,
                            shrinkA=0, shrinkB=0)
    ax.add_patch(arr)


def draw_box(x, y, w, h, title, subtitle=None, edgecolor=C_EDGE, fill=C_FILL,
             lw=LW_BOX, title_fs=TITLE_FS, sub_fs=SUB_FS, ls='-'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                           linewidth=lw, edgecolor=edgecolor, facecolor=fill,
                           linestyle=ls, zorder=3)
    ax.add_patch(box)
    cx = x + w / 2
    if subtitle:
        ax.text(cx, y + h * 0.66, title, ha='center', va='center',
                 fontsize=title_fs, color=C_TEXT, zorder=4)
        ax.text(cx, y + h * 0.27, subtitle, ha='center', va='center',
                 fontsize=sub_fs, color=C_SUB, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha='center', va='center',
                 fontsize=title_fs, color=C_TEXT, zorder=4)


def tick_strip(x0, x1, y, h, n_ticks, highlight_idx=None):
    """A horizontal strip with light tick marks for discrete time steps;
    optionally highlights one tick (returns its centre x)."""
    bar = FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                           boxstyle='round,pad=0.01,rounding_size=0.05',
                           linewidth=1.1, edgecolor=C_EDGE, facecolor=C_FILL, zorder=3)
    ax.add_patch(bar)
    edges = np.linspace(x0, x1, n_ticks + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for i, xc in enumerate(centres):
        if highlight_idx is not None and i == highlight_idx:
            continue
        ax.plot([xc, xc], [y - h * 0.32, y + h * 0.32], color=C_SUB, lw=0.8, zorder=4)
    hi_x = None
    if highlight_idx is not None:
        hi_x = centres[highlight_idx]
        ax.plot([hi_x, hi_x], [y - h * 0.42, y + h * 0.42], color=C_AGG, lw=2.2, zorder=5)
    return hi_x


def mini_square(x, y, s, color=C_EDGE, fill='white', lw=1.0, zorder=3):
    box = FancyBboxPatch((x, y), s, s, boxstyle='square,pad=0',
                           linewidth=lw, edgecolor=color, facecolor=fill, zorder=zorder)
    ax.add_patch(box)


def grid_matrix(x, y, w, h, rows, cols, row_colors=None, edge=C_EDGE, lw=1.2):
    box = FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0',
                           linewidth=lw, edgecolor=edge, facecolor='white', zorder=3)
    ax.add_patch(box)
    if row_colors and rows == 2:
        ax.add_patch(Rectangle((x, y + h / 2), w, h / 2, facecolor=row_colors[0],
                                 alpha=0.14, lw=0, zorder=2))
        ax.add_patch(Rectangle((x, y), w, h / 2, facecolor=row_colors[1],
                                 alpha=0.14, lw=0, zorder=2))
    for c in range(1, cols):
        xc = x + w * c / cols
        ax.plot([xc, xc], [y, y + h], color=edge, lw=0.35, alpha=0.55, zorder=3)
    for r in range(1, rows):
        yr = y + h * r / rows
        ax.plot([x, x + w], [yr, yr], color=edge, lw=0.6, alpha=0.7, zorder=3)


cursor = 0.4  # running x-cursor; every stage advances it by its own true content width

# =============================================================================
# STAGE 1 — equilibrium segment
# =============================================================================

S1_X0 = cursor
S1_W = 2.6
S1_X1 = S1_X0 + S1_W
section_header((S1_X0 + S1_X1) / 2, 'Equilibrium segment')

hi_x = tick_strip(S1_X0, S1_X1, Y_MID, 0.55, n_ticks=16, highlight_idx=6)
ax.text((S1_X0 + S1_X1) / 2, Y_MID - 0.55, 'contiguous equilibrium-labelled\nraw time steps',
         ha='center', va='top', fontsize=NOTE_FS, color=C_SUB, style='italic')

cursor = S1_X1 + STAGE_GAP
divider(cursor - STAGE_GAP / 2)

# =============================================================================
# STAGE 2 — per-time-step G5 feature matrix (callout from the highlighted tick)
# =============================================================================

LABEL_MARGIN = 0.55
MAT_W, MAT_H = 1.85, 1.0
S2_X0 = cursor
mat_x = S2_X0 + LABEL_MARGIN
mat_y = Y_MID + 0.35
grid_matrix(mat_x, mat_y, MAT_W, MAT_H, rows=2, cols=9, row_colors=(C_UP, C_DOWN))
ax.text(mat_x - 0.12, mat_y + MAT_H * 0.75, 'up', ha='right', va='center',
         fontsize=NOTE_FS, color=C_UP, fontweight='bold')
ax.text(mat_x - 0.12, mat_y + MAT_H * 0.25, 'down', ha='right', va='center',
         fontsize=NOTE_FS, color=C_DOWN, fontweight='bold')
ax.text(mat_x + MAT_W / 2, mat_y + MAT_H + 0.45, r'$X_t \in \mathbb{R}^{2\times 9}$',
         ha='center', va='bottom', fontsize=TITLE_FS + 0.5)
ax.text(mat_x + MAT_W / 2, mat_y + MAT_H + 0.15, 'G5: 9 normalized intervals per direction',
         ha='center', va='bottom', fontsize=NOTE_FS, color=C_SUB, style='italic')
ax.text(mat_x + MAT_W / 2, mat_y - 0.18, 'one raw time step',
         ha='center', va='top', fontsize=NOTE_FS, color=C_SUB)

straight_arrow((hi_x, Y_MID + 0.30), (mat_x + MAT_W * 0.15, mat_y - 0.05),
                color=C_AGG, lw=1.0, mscale=9, ls='--')

S2_X1 = mat_x + MAT_W + 0.25
cursor = S2_X1 + STAGE_GAP
divider(cursor - STAGE_GAP / 2)

# =============================================================================
# STAGE 3 — temporal aggregation (mean)
# =============================================================================

S3_X0 = cursor
lx = S3_X0

raw_y = Y_MID + 0.15
raw_s, raw_gap = 0.26, 0.08
n_raw = 5
for i in range(n_raw):
    mini_square(lx + i * (raw_s + raw_gap), raw_y, raw_s, color=C_SUB, fill='white')
raw_x1 = lx + n_raw * (raw_s + raw_gap) - raw_gap
ax.text((lx + raw_x1) / 2, raw_y - 0.22, r'16 raw steps $X_t$',
         ha='center', va='top', fontsize=NOTE_FS, color=C_SUB)
ax.plot([lx - 0.05, raw_x1 + 0.05], [raw_y + raw_s + 0.10, raw_y + raw_s + 0.10],
         color=C_SUB, lw=0.8)
lx = raw_x1

lx += 0.20
arrow_x0 = lx
lx += 0.45
agg_x = lx
agg_y = raw_y
agg_s = 0.36
straight_arrow((arrow_x0, raw_y + raw_s / 2), (agg_x - 0.05, agg_y + agg_s / 2),
                color=C_AGG, mscale=11)
ax.text((arrow_x0 + agg_x) / 2, raw_y + raw_s / 2 + 0.20, 'mean',
         ha='center', va='bottom', fontsize=NOTE_FS + 0.3, color=C_AGG, fontweight='bold')
mini_square(agg_x, agg_y, agg_s, color=C_AGG, fill='white', lw=1.8)
ax.text(agg_x + agg_s / 2, agg_y - 0.18, r'$\bar X_k$', ha='center', va='top',
         fontsize=NOTE_FS + 1, color=C_AGG)
lx = agg_x + agg_s

lx += 0.40
seq_x0 = lx
seq_s, seq_gap = 0.22, 0.06
n_seq = 6
for i in range(n_seq):
    mini_square(seq_x0 + i * (seq_s + seq_gap), agg_y + 0.05, seq_s, color=C_AGG,
                 fill='white', lw=1.0)
seq_x1 = seq_x0 + n_seq * (seq_s + seq_gap) - seq_gap
ax.text((seq_x0 + seq_x1) / 2, agg_y - 0.20,
         r'aggregated sequence $\{\bar X_k\}$', ha='center', va='top',
         fontsize=NOTE_FS, color=C_SUB)
lx = seq_x1

S3_X1 = lx
section_header((S3_X0 + S3_X1) / 2, 'Temporal aggregation')
ax.text((S3_X0 + S3_X1) / 2, Y_MID - 0.85,
         r'$\bar X_k = \mathrm{mean}(X_{16k},\ldots,X_{16k+15})$',
         ha='center', va='top', fontsize=MATH_FS - 0.5, color=C_SUB)

straight_arrow((mat_x + MAT_W + 0.05, mat_y + MAT_H / 2), (S3_X0 - 0.08, raw_y + raw_s / 2),
                color=C_EDGE)

cursor = S3_X1 + STAGE_GAP
divider(cursor - STAGE_GAP / 2)
seq_end_y = agg_y + 0.05 + seq_s / 2

# =============================================================================
# STAGE 4 — patch extraction (train vs val/test)
# =============================================================================

S4_X0 = cursor

row_s = 0.26
n_row = 16
win_w_cells = 8   # W=64 shown abstractly as 8 aggregated cells

def cell_row(x0, y, n, s, color=C_SUB):
    for i in range(n):
        mini_square(x0 + i * s, y, s, color=color, fill='white', lw=0.7, zorder=3)


def window_overlay(x0, y, s, start_cells, win_cells, color, ls='-'):
    xw = x0 + start_cells * s
    ww = win_cells * s
    box = FancyBboxPatch((xw, y - 0.06), ww, s + 0.12,
                           boxstyle='round,pad=0.0,rounding_size=0.03',
                           linewidth=1.4, edgecolor=color, facecolor=color,
                           alpha=0.16, linestyle=ls, zorder=5)
    ax.add_patch(box)
    edge_only = FancyBboxPatch((xw, y - 0.06), ww, s + 0.12,
                                 boxstyle='round,pad=0.0,rounding_size=0.03',
                                 linewidth=1.4, edgecolor=color, facecolor='none',
                                 linestyle=ls, zorder=6)
    ax.add_patch(edge_only)


row_w = n_row * row_s

# --- training: overlapping windows, stride = 25% of W (75% overlap) ---
train_y = Y_MID + 1.05
cell_row(S4_X0, train_y, n_row, row_s)
for start in (0, 2, 4, 6, 8):
    window_overlay(S4_X0, train_y, row_s, start, win_w_cells, C_TRAIN, ls='-')
ax.text(S4_X0 + row_w / 2, train_y + row_s + 0.55,
         'Training: 75% overlap', ha='center', va='bottom',
         fontsize=TITLE_FS - 0.5, color=C_TRAIN, fontweight='bold')
ax.text(S4_X0 + row_w / 2, train_y + row_s + 0.28,
         r'stride $= W(1-0.75) = 16$', ha='center', va='bottom',
         fontsize=NOTE_FS, color=C_SUB, style='italic')

# --- val/test: non-overlapping windows, stride = W ---
eval_y = Y_MID - 0.95
cell_row(S4_X0, eval_y, n_row, row_s)
for start in (0, 8):
    window_overlay(S4_X0, eval_y, row_s, start, win_w_cells, C_EVAL, ls='--')
ax.text(S4_X0 + row_w / 2, eval_y - 0.28,
         'Validation / Test: no overlap', ha='center', va='top',
         fontsize=TITLE_FS - 0.5, color=C_EVAL, fontweight='bold')
ax.text(S4_X0 + row_w / 2, eval_y - 0.55,
         r'stride $= W = 64$', ha='center', va='top',
         fontsize=NOTE_FS, color=C_SUB, style='italic')

S4_X1 = S4_X0 + row_w + 1.7

straight_arrow((seq_x1 + 0.05, seq_end_y), (S4_X0 - 0.08, train_y + row_s / 2),
                color=C_TRAIN, lw=1.0, mscale=10)
straight_arrow((seq_x1 + 0.05, seq_end_y), (S4_X0 - 0.08, eval_y + row_s / 2),
                color=C_EVAL, lw=1.0, mscale=10, ls='--')

section_header((S4_X0 + S4_X0 + row_w) / 2, 'Patch extraction')

cursor = S4_X1 + STAGE_GAP
divider(cursor - STAGE_GAP / 2)

# =============================================================================
# STAGE 5 — final patch tensor
# =============================================================================

S5_X0 = cursor

t_w, t_h = 1.9, 0.85
back_x, back_y = S5_X0 + 0.32, Y_MID - 0.10
front_x, front_y = S5_X0, Y_MID - 0.42

back = FancyBboxPatch((back_x, back_y), t_w, t_h, boxstyle='square,pad=0',
                        linewidth=1.3, edgecolor=C_DOWN, facecolor='white', zorder=3)
ax.add_patch(back)
for r in range(1, 4):
    yr = back_y + t_h * r / 4
    ax.plot([back_x, back_x + t_w], [yr, yr], color=C_DOWN, lw=0.4, alpha=0.6, zorder=4)

front = FancyBboxPatch((front_x, front_y), t_w, t_h, boxstyle='square,pad=0',
                         linewidth=1.3, edgecolor=C_UP, facecolor='white', zorder=5)
ax.add_patch(front)
for r in range(1, 4):
    yr = front_y + t_h * r / 4
    ax.plot([front_x, front_x + t_w], [yr, yr], color=C_UP, lw=0.4, alpha=0.6, zorder=6)

ax.text(front_x + t_w / 2, front_y - 0.20, '64 time steps', ha='center', va='top',
         fontsize=NOTE_FS, color=C_SUB)
ax.text(back_x + t_w + 0.15, back_y + t_h * 0.5, '9\nfeatures', ha='left', va='center',
         fontsize=NOTE_FS, color=C_SUB)

S5_X1 = back_x + t_w + 1.1
ax.text((front_x + S5_X1) / 2, back_y + t_h + 1.0, r'$x \in \mathbb{R}^{2\times 9\times 64}$',
         ha='center', va='bottom', fontsize=TITLE_FS + 1)
ax.text((front_x + S5_X1) / 2, back_y + t_h + 0.45, 'channels = 2 (up / down)',
         ha='center', va='bottom', fontsize=NOTE_FS, color=C_SUB)

section_header((front_x + S5_X1) / 2, 'Final patch tensor')

straight_arrow((S4_X1 - 1.55, (train_y + eval_y) / 2 + row_s / 2),
                (front_x - 0.15, Y_MID), color=C_EDGE)

cursor = S5_X1 + 0.4
FIG_W = cursor
fig.set_size_inches(FIG_W, FIG_H)  # match physical size to final content extent
ax.set_xlim(0, FIG_W)              # so 1 data-unit = 1 inch, as every constant assumes
ax.set_ylim(0, FIG_H)

# =============================================================================
# SAVE
# =============================================================================

out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
for ext, dpi in [('pdf', None), ('png', 300)]:
    p = out_dir / f'feature_patch_construction.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved: {p}')
plt.close(fig)

print('\n=== Patch construction — reproducibility report ===')
print('  Verified against configs/preparation/data_preparation.json and')
print('  src/preparation/data_preparation.py::_prepare_patches:')
print('    downsample_factor (block mean size):     16')
print('    patch_width W:                           64')
print('    patch_overlap (train only):               0.75  ->  stride 16')
print('    val/test overlap (forced):                0.0   ->  stride 64')
print('  Verified against configs/contrastive/encoder.json:')
print('    selected_features: 18 plain G5 columns (norm_interval_up/down_0..8)')
print('    cnn_channel_mode: dual  ->  reshape_to_channels(n_stats=1) -> (2, 9, W)')
print('  Final patch tensor: x in R^(2 x 9 x 64)')
