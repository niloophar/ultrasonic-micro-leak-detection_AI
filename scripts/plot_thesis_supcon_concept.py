#!/usr/bin/env python
"""
Thesis figure -- conceptual illustration of the supervised contrastive
(SupCon) mechanism used to train the contrastive encoder.

NOT based on real experimental embeddings -- purely schematic, for the
Methods chapter.

Two panels:
  (a) Series-stratified mini-batch: samples from three recording series
      (color), each series containing both classes (shape). One anchor
      is highlighted; same-class cross-series pairs are marked positive,
      opposite-class pairs (same- or cross-series) are marked negative.
  (b) Supervised contrastive objective: a conceptual 2D projection where
      positives are pulled toward the anchor and negatives are pushed
      away, so the space organizes by class while staying mixed across
      series.

Outputs:
  figures/supcon_concept.pdf
  figures/supcon_concept.png (600 dpi)

Run: python scripts/plot_thesis_supcon_concept.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle
from matplotlib.lines import Line2D

# ─── palette ────────────────────────────────────────────────────────────────
C_TEXT = '#232323'
C_SUB = '#5c5c5c'
C_MUTED = '#8f8f8f'

C_SERIES_A = '#5b8fc2'   # blue
C_SERIES_B = '#4f9a8f'   # teal
C_SERIES_C = '#c9903f'   # amber
SERIES_COLORS = {'A': C_SERIES_A, 'B': C_SERIES_B, 'C': C_SERIES_C}

C_MARKER_EDGE = '#333333'
C_ANCHOR_RING = '#b8860b'
C_POS_LINE = '#3f8f7a'
C_NEG_LINE = '#c1663c'

C_PANEL_EDGE = '#9a9a9a'
C_PANEL_FILL = '#fbfbfa'
C_BLOB_NF_EDGE = '#7ea3c4'
C_BLOB_NF_FILL = '#eef3f9'
C_BLOB_LK_EDGE = '#cf9a7a'
C_BLOB_LK_FILL = '#fbf2ec'

LW_PANEL = 1.2
MARKER_SIZE = 240

TITLE_FS = 15.5
SUB_FS = 10.5
LABEL_FS = 10.5
SMALL_FS = 9.0
TINY_FS = 8.3

# ─── figure setup ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 15.5, 8.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')


# ─── helpers ─────────────────────────────────────────────────────────────────
def draw_panel_box(x, y, w, h, dashed=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.22',
        linewidth=LW_PANEL, edgecolor=C_PANEL_EDGE, facecolor=C_PANEL_FILL,
        linestyle=(0, (5, 3)) if dashed else '-', zorder=0,
    )
    ax.add_patch(box)


def draw_marker(x, y, shape, color, size=MARKER_SIZE, zorder=5):
    m = 'o' if shape == 'circle' else '^'
    ax.scatter([x], [y], s=size, marker=m, facecolor=color,
               edgecolor=C_MARKER_EDGE, linewidth=1.0, zorder=zorder)


def draw_anchor_ring(x, y, r=0.34, zorder=4):
    ring = Circle((x, y), r, fill=False, edgecolor=C_ANCHOR_RING,
                  linewidth=1.8, linestyle=(0, (3, 1.5)), zorder=zorder)
    ax.add_patch(ring)


def draw_connector(p_from, p_to, color, curve=0.0, zorder=2, lw=1.3):
    arr = FancyArrowPatch(
        p_from, p_to, arrowstyle='-', linewidth=lw, color=color,
        linestyle=(0, (4, 2.5)),
        connectionstyle=f'arc3,rad={curve}', zorder=zorder,
        shrinkA=8, shrinkB=8,
    )
    ax.add_patch(arr)


def draw_force_arrow(p_from, p_to, color, curve=0.0, zorder=6, lw=1.6, mscale=11):
    arr = FancyArrowPatch(
        p_from, p_to, arrowstyle='-|>', mutation_scale=mscale,
        linewidth=lw, color=color, connectionstyle=f'arc3,rad={curve}',
        zorder=zorder, shrinkA=6, shrinkB=6,
    )
    ax.add_patch(arr)


# offset pattern shared by all three series clusters in panel (a):
# (dx, dy, shape) -- identical layout per series, only recolored/translated
CLUSTER_OFFSETS = [
    (-0.55, 1.30, 'circle'),    # 0
    (0.55, 1.05, 'circle'),     # 1
    (-0.15, 0.10, 'triangle'),  # 2
    (0.60, -0.40, 'circle'),    # 3
    (-0.55, -0.95, 'triangle'), # 4
    (0.20, -1.30, 'triangle'),  # 5
]

# ═════════════════════════════════════════════════════════════════════════
# PANEL (a) — SERIES-STRATIFIED MINI-BATCH
# ═════════════════════════════════════════════════════════════════════════
PA_X, PA_Y, PA_W, PA_H = 0.4, 1.05, 7.0, 6.05
draw_panel_box(PA_X, PA_Y, PA_W, PA_H)
ax.text(PA_X + PA_W / 2, 7.55, '(a) Series-stratified mini-batch',
         ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

CENTERS_A = {'A': (1.85, 4.15), 'B': (3.90, 4.15), 'C': (5.95, 4.15)}
points_a = {}  # series -> list of (x,y,shape)

for s, (cx, cy) in CENTERS_A.items():
    ax.text(cx, cy + 2.10, f'Series {s}', ha='center', va='bottom',
             fontsize=LABEL_FS, color=SERIES_COLORS[s], fontweight='bold')
    pts = []
    for dx, dy, shape in CLUSTER_OFFSETS:
        px, py = cx + dx, cy + dy
        pts.append((px, py, shape))
    points_a[s] = pts

# draw all markers first (so connector lines sit under/over cleanly)
for s, pts in points_a.items():
    for px, py, shape in pts:
        draw_marker(px, py, shape, SERIES_COLORS[s], zorder=5)

# anchor = Series A, point index 1 (a NoFlow circle)
anchor_x, anchor_y, _ = points_a['A'][1]

# positive pairs: same class (circle), cross-series -> B[1], C[1]
pos_targets = [points_a['B'][1], points_a['C'][1]]
for px, py, _ in pos_targets:
    draw_connector((anchor_x, anchor_y), (px, py), C_POS_LINE, curve=-0.12)

# negative pairs: opposite class (triangle) -- one same-series, one cross-series
neg_targets = [points_a['A'][2], points_a['B'][4]]
for px, py, _ in neg_targets:
    draw_connector((anchor_x, anchor_y), (px, py), C_NEG_LINE, curve=0.12)

draw_anchor_ring(anchor_x, anchor_y)
ax.annotate('anchor', xy=(anchor_x - 0.28, anchor_y + 0.30),
             xytext=(anchor_x - 1.55, anchor_y + 1.55),
             fontsize=SMALL_FS, color=C_TEXT, fontweight='bold',
             ha='center',
             arrowprops=dict(arrowstyle='-', color=C_ANCHOR_RING, lw=1.1,
                              connectionstyle='arc3,rad=0.15'))

# small inline pos/neg labels near a representative connector, offset clear
# of the fixed marker grid (checked against CLUSTER_OFFSETS positions)
pos_label_x = anchor_x + 0.30 * (pos_targets[0][0] - anchor_x)
pos_label_y = anchor_y - 0.35
ax.text(pos_label_x, pos_label_y, 'positive', fontsize=TINY_FS,
         color=C_POS_LINE, style='italic', ha='center')
mid_neg = ((anchor_x + neg_targets[1][0]) / 2, (anchor_y + neg_targets[1][1]) / 2)
ax.text(mid_neg[0] + 0.10, mid_neg[1] - 0.32, 'negative', fontsize=TINY_FS,
         color=C_NEG_LINE, style='italic', ha='center')

ax.text(PA_X + PA_W / 2, PA_Y + 0.20,
         'same class, different series $\\rightarrow$ positive   |   '
         'different class $\\rightarrow$ negative',
         ha='center', va='bottom', fontsize=TINY_FS, color=C_MUTED, style='italic')

# ═════════════════════════════════════════════════════════════════════════
# PANEL (b) — SUPERVISED CONTRASTIVE OBJECTIVE (projected latent space)
# ═════════════════════════════════════════════════════════════════════════
PB_X, PB_Y, PB_W, PB_H = 8.1, 1.05, 7.0, 6.05
draw_panel_box(PB_X, PB_Y, PB_W, PB_H)
ax.text(PB_X + PB_W / 2, 7.55, '(b) Supervised contrastive objective',
         ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

NF_CX, NF_CY = 10.05, 4.20
LK_CX, LK_CY = 13.45, 4.20
BLOB_RX, BLOB_RY = 1.55, 2.35

ax.add_patch(Ellipse((NF_CX, NF_CY), width=2 * BLOB_RX, height=2 * BLOB_RY,
                       facecolor=C_BLOB_NF_FILL, edgecolor=C_BLOB_NF_EDGE,
                       linewidth=1.1, linestyle=(0, (4, 2)), zorder=1))
ax.add_patch(Ellipse((LK_CX, LK_CY), width=2 * BLOB_RX, height=2 * BLOB_RY,
                       facecolor=C_BLOB_LK_FILL, edgecolor=C_BLOB_LK_EDGE,
                       linewidth=1.1, linestyle=(0, (4, 2)), zorder=1))

ax.text(NF_CX, NF_CY + BLOB_RY + 0.28, 'NoFlow  $\\bigcirc$', ha='center', va='bottom',
         fontsize=LABEL_FS, color='#3a5f80', fontweight='bold')
ax.text(LK_CX, LK_CY + BLOB_RY + 0.28, 'Leakage  $\\triangle$', ha='center', va='bottom',
         fontsize=LABEL_FS, color='#a55b34', fontweight='bold')

# scattered points inside each blob, mixed series colors -> "mixed across series"
nf_layout = [  # (dx, dy, series)
    (-0.65, 1.35, 'B'), (0.15, 1.55, 'C'), (0.70, 0.75, 'A'),
    (-0.85, 0.35, 'C'), (0.05, -0.15, 'B'), (0.80, -0.55, 'A'),
    (-0.55, -1.05, 'A'), (0.15, -1.55, 'C'), (-0.05, 0.95, 'A'),
]
lk_layout = [
    (-0.60, 1.30, 'C'), (0.20, 1.55, 'A'), (0.75, 0.65, 'B'),
    (-0.80, 0.30, 'A'), (0.00, -0.20, 'C'), (0.75, -0.60, 'B'),
    (-0.55, -1.10, 'B'), (0.20, -1.55, 'A'), (-0.10, 0.90, 'C'),
]

nf_points = []
for dx, dy, s in nf_layout:
    px, py = NF_CX + dx, NF_CY + dy
    nf_points.append((px, py, s))
    draw_marker(px, py, 'circle', SERIES_COLORS[s], size=190, zorder=5)

lk_points = []
for dx, dy, s in lk_layout:
    px, py = LK_CX + dx, LK_CY + dy
    lk_points.append((px, py, s))
    draw_marker(px, py, 'triangle', SERIES_COLORS[s], size=190, zorder=5)

# anchor: the Series-A NoFlow point closest to the blob boundary (index 8, series A)
b_anchor_x, b_anchor_y, _ = nf_points[8]
draw_anchor_ring(b_anchor_x, b_anchor_y, r=0.30)
ax.annotate('anchor', xy=(b_anchor_x - 0.05, b_anchor_y + 0.28),
             xytext=(b_anchor_x - 1.05, b_anchor_y + 1.35),
             fontsize=SMALL_FS, color=C_TEXT, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='-', color=C_ANCHOR_RING, lw=1.1,
                              connectionstyle='arc3,rad=0.15'))

# pull arrows: two other-series NoFlow points -> anchor
pull_sources = [nf_points[0], nf_points[1]]
for px, py, _ in pull_sources:
    draw_force_arrow((px, py), (b_anchor_x, b_anchor_y), C_POS_LINE, curve=-0.08)

# push arrows: two Leakage points near the boundary, pushed further away from anchor
push_idx = [2, 5]
for i in push_idx:
    px, py, _ = lk_points[i]
    dxp, dyp = px - b_anchor_x, py - b_anchor_y
    norm = (dxp ** 2 + dyp ** 2) ** 0.5
    ux, uy = dxp / norm, dyp / norm
    start = (px - 0.35 * ux, py - 0.35 * uy)
    end = (px + 0.45 * ux, py + 0.45 * uy)
    draw_force_arrow(start, end, C_NEG_LINE, curve=0.0)

ax.text(PB_X + PB_W / 2, PB_Y + 0.20,
         'organized by operating state, not by recording series',
         ha='center', va='bottom', fontsize=TINY_FS, color=C_MUTED, style='italic')

# ═════════════════════════════════════════════════════════════════════════
# SHARED LEGEND
# ═════════════════════════════════════════════════════════════════════════
LEG_Y = 0.42

def legend_swatch_square(x, y, color, s=0.20):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor=color,
                             edgecolor=C_MARKER_EDGE, linewidth=0.8, zorder=6))

def legend_text(x, y, text, color=C_TEXT, fs=SMALL_FS, weight='normal'):
    ax.text(x, y, text, ha='left', va='center', fontsize=fs, color=color,
             fontweight=weight)

x = 0.5
legend_text(x, LEG_Y, 'Series (color):', fs=SMALL_FS, weight='bold')
x += 1.55
for s in ['A', 'B', 'C']:
    legend_swatch_square(x, LEG_Y, SERIES_COLORS[s])
    legend_text(x + 0.20, LEG_Y, s, fs=SMALL_FS)
    x += 0.62

x += 0.55
legend_text(x, LEG_Y, 'Class (shape):', fs=SMALL_FS, weight='bold')
x += 1.65
ax.scatter([x], [LEG_Y], s=150, marker='o', facecolor='#dcdcdc',
           edgecolor=C_MARKER_EDGE, linewidth=0.9, zorder=6)
legend_text(x + 0.28, LEG_Y, 'NoFlow', fs=SMALL_FS)
x += 1.55
ax.scatter([x], [LEG_Y], s=150, marker='^', facecolor='#dcdcdc',
           edgecolor=C_MARKER_EDGE, linewidth=0.9, zorder=6)
legend_text(x + 0.28, LEG_Y, 'Leakage', fs=SMALL_FS)

x += 1.75
ax.add_line(Line2D([x, x + 0.55], [LEG_Y, LEG_Y], color=C_POS_LINE, lw=1.6,
                     linestyle=(0, (4, 2.5))))
legend_text(x + 0.72, LEG_Y, 'positive (pull)', color=C_POS_LINE, fs=SMALL_FS)
x += 2.55
ax.add_line(Line2D([x, x + 0.55], [LEG_Y, LEG_Y], color=C_NEG_LINE, lw=1.6,
                     linestyle=(0, (4, 2.5))))
legend_text(x + 0.72, LEG_Y, 'negative (push)', color=C_NEG_LINE, fs=SMALL_FS)

x += 2.60
ring = Circle((x + 0.10, LEG_Y), 0.16, fill=False, edgecolor=C_ANCHOR_RING,
              linewidth=1.6, linestyle=(0, (3, 1.5)))
ax.add_patch(ring)
legend_text(x + 0.40, LEG_Y, 'anchor', color=C_TEXT, fs=SMALL_FS)

# ─── save ───────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 600)]:
    p = out_dir / f'supcon_concept.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
