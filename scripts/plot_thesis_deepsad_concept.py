#!/usr/bin/env python
"""
Thesis figure -- conceptual illustration of the DeepSAD one-class
objective used to train the contrastive encoder's DeepSAD variant.

NOT based on real experimental embeddings -- purely schematic, for the
Methods chapter. Shares its visual vocabulary (palette, marker/shape
conventions, panel style, legend layout) with the SupCon concept figure
(plot_thesis_supcon_concept.py).

Two panels:
  (a) Initial projected space: NoFlow (circles) dispersed around the
      fixed center c, several Leakage samples (triangles) intruding
      inside the margin m. Faint arrows indicate the intended
      optimization direction.
  (b) DeepSAD-structured space: NoFlow contracted into a compact region
      near c; Leakage pushed outside the margin without being required
      to form a compact cluster of its own. c and m are unchanged.

Outputs:
  figures/deepsad_concept.pdf
  figures/deepsad_concept.png (600 dpi)

Run: python scripts/plot_thesis_deepsad_concept.py
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

# ─── palette (shared with plot_thesis_supcon_concept.py) ────────────────────
C_TEXT = '#232323'
C_SUB = '#5c5c5c'
C_MUTED = '#8f8f8f'

C_SERIES_A = '#5b8fc2'   # blue
C_SERIES_B = '#4f9a8f'   # teal
C_SERIES_C = '#c9903f'   # amber
SERIES_COLORS = {'A': C_SERIES_A, 'B': C_SERIES_B, 'C': C_SERIES_C}
SERIES_LIST = ['A', 'B', 'C']

C_MARKER_EDGE = '#333333'
C_CENTER = '#b8860b'      # fixed center c -- same gold as the SupCon anchor ring
C_MARGIN = '#a9825a'
C_POS_LINE = '#3f8f7a'    # contraction / pull semantics
C_NEG_LINE = '#c1663c'    # margin-penalty / push semantics

C_PANEL_EDGE = '#9a9a9a'
C_PANEL_FILL = '#fbfbfa'
C_FORMULA_FILL = '#fff7e2'
C_FORMULA_EDGE = '#b8903f'

LW_PANEL = 1.2
MARKER_SIZE = 220

TITLE_FS = 15.5
LABEL_FS = 10.5
SMALL_FS = 9.0
TINY_FS = 8.3

RNG = np.random.default_rng(42)

# ─── figure setup ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 15.5, 8.6
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')


# ─── helpers ─────────────────────────────────────────────────────────────────
def draw_panel_box(x, y, w, h):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.22',
        linewidth=LW_PANEL, edgecolor=C_PANEL_EDGE, facecolor=C_PANEL_FILL,
        linestyle=(0, (5, 3)), zorder=0,
    )
    ax.add_patch(box)


def draw_marker(x, y, shape, color, size=MARKER_SIZE, zorder=5):
    m = 'o' if shape == 'circle' else '^'
    ax.scatter([x], [y], s=size, marker=m, facecolor=color,
               edgecolor=C_MARKER_EDGE, linewidth=1.0, zorder=zorder)


def draw_force_arrow(p_from, p_to, color, curve=0.0, zorder=3, lw=1.3,
                      mscale=9, alpha=1.0):
    arr = FancyArrowPatch(
        p_from, p_to, arrowstyle='-|>', mutation_scale=mscale,
        linewidth=lw, color=color, alpha=alpha,
        connectionstyle=f'arc3,rad={curve}',
        zorder=zorder, shrinkA=5, shrinkB=5,
    )
    ax.add_patch(arr)


def sample_ring(n, r_lo, r_hi, ang_lo=0.0, ang_hi=2 * np.pi):
    """Deterministic (fixed-seed RNG) points in a radius band and angle range."""
    radii = RNG.uniform(r_lo, r_hi, n)
    angles = RNG.uniform(ang_lo, ang_hi, n)
    series = RNG.choice(SERIES_LIST, size=n)
    return radii, angles, series


def place(cx, cy, radii, angles, series, shape):
    """Draw markers from polar offsets around (cx, cy); return abs coords."""
    pts = []
    for r, a, s in zip(radii, angles, series):
        px, py = cx + r * np.cos(a), cy + r * np.sin(a)
        draw_marker(px, py, shape, SERIES_COLORS[s])
        pts.append((px, py, s))
    return pts


def draw_center_and_margin(cx, cy, m, label_angle=42.0):
    circ = Circle((cx, cy), m, fill=False, edgecolor=C_MARGIN, linewidth=1.4,
                  linestyle=(0, (4, 2.5)), zorder=2)
    ax.add_patch(circ)
    a = np.deg2rad(label_angle)
    tx, ty = cx + (m + 0.30) * np.cos(a), cy + (m + 0.30) * np.sin(a)
    ax.text(tx, ty, '$m$', ha='center', va='center', fontsize=LABEL_FS,
             color=C_MARGIN, fontweight='bold')
    ax.scatter([cx], [cy], s=520, marker='*', facecolor=C_CENTER,
               edgecolor='#7a5c0a', linewidth=0.9, zorder=7)
    ax.text(cx + 0.30, cy + 0.34, r'$\mathbf{c}$', ha='left', va='bottom',
             fontsize=LABEL_FS + 1, color=C_TEXT, fontweight='bold', zorder=7)


M = 1.55  # shared margin radius, identical in both panels

# ═════════════════════════════════════════════════════════════════════════
# PANEL (a) — INITIAL PROJECTED SPACE
# ═════════════════════════════════════════════════════════════════════════
PA_X, PA_Y, PA_W, PA_H = 0.4, 2.05, 7.0, 5.75
draw_panel_box(PA_X, PA_Y, PA_W, PA_H)
ax.text(PA_X + PA_W / 2, 7.95, '(a) Initial projected space',
         ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

CA_X, CA_Y = PA_X + PA_W / 2, PA_Y + PA_H / 2 + 0.15  # (3.9, 4.90)

# NoFlow: dispersed around c, straddling the margin (not yet contracted).
# Kept clear of a small no-go radius around c so the star stays legible.
r, a, s = sample_ring(10, 0.75, 2.30)
nf_before = place(CA_X, CA_Y, r, a, s, 'circle')

# Leakage: a few intruding inside the margin, the rest outside
r_in, a_in, s_in = sample_ring(4, 0.70, 1.15)
lk_before_in = place(CA_X, CA_Y, r_in, a_in, s_in, 'triangle')
r_out, a_out, s_out = sample_ring(5, 1.85, 2.55, ang_lo=np.pi * 0.2, ang_hi=np.pi * 2.0)
lk_before_out = place(CA_X, CA_Y, r_out, a_out, s_out, 'triangle')

draw_center_and_margin(CA_X, CA_Y, M)

# faint arrows: intended optimization direction. Sources chosen as the
# largest-radius NoFlow points (visible pull toward c without crowding
# the star) and the Leakage points already inside the margin (pushed out).
nf_by_radius = sorted(nf_before, key=lambda p: -((p[0] - CA_X) ** 2 + (p[1] - CA_Y) ** 2))
for px, py, _ in nf_by_radius[:3]:
    dx, dy = CA_X - px, CA_Y - py
    norm = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / norm, dy / norm
    draw_force_arrow((px + 0.20 * ux, py + 0.20 * uy),
                      (px + 0.90 * ux, py + 0.90 * uy),
                      C_POS_LINE, alpha=0.45, lw=1.1, mscale=7, zorder=2)

for px, py, _ in lk_before_in:
    dx, dy = px - CA_X, py - CA_Y
    norm = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / norm, dy / norm
    draw_force_arrow((px + 0.20 * ux, py + 0.20 * uy),
                      (px + 0.85 * ux, py + 0.85 * uy),
                      C_NEG_LINE, alpha=0.45, lw=1.1, mscale=7, zorder=2)

ax.text(PA_X + PA_W / 2, PA_Y + 0.20,
         'faint arrows: intended optimization direction',
         ha='center', va='bottom', fontsize=TINY_FS, color=C_MUTED, style='italic')

# ═════════════════════════════════════════════════════════════════════════
# PANEL (b) — DEEPSAD-STRUCTURED SPACE
# ═════════════════════════════════════════════════════════════════════════
PB_X, PB_Y, PB_W, PB_H = 8.1, 2.05, 7.0, 5.75
draw_panel_box(PB_X, PB_Y, PB_W, PB_H)
ax.text(PB_X + PB_W / 2, 7.95, '(b) DeepSAD-structured space',
         ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

CB_X, CB_Y = PB_X + PB_W / 2, PB_Y + PB_H / 2 + 0.15  # (11.6, 4.90)

# NoFlow: contracted into a compact region close to c
r, a, s = sample_ring(10, 0.05, 0.50)
nf_after = place(CB_X, CB_Y, r, a, s, 'circle')

# Leakage: outside the margin, spread widely in radius & angle -- no single cluster
r, a, s = sample_ring(9, 1.75, 2.70)
lk_after = place(CB_X, CB_Y, r, a, s, 'triangle')

draw_center_and_margin(CB_X, CB_Y, M)

# "NoFlow contraction" annotation
ax.annotate('NoFlow\ncontraction', xy=(CB_X - 0.30, CB_Y + 0.30),
             xytext=(CB_X - 2.55, CB_Y + 2.15),
             fontsize=SMALL_FS, color=C_POS_LINE, fontweight='bold', ha='center',
             linespacing=1.3,
             arrowprops=dict(arrowstyle='-|>', color=C_POS_LINE, lw=1.2,
                              connectionstyle='arc3,rad=-0.15',
                              mutation_scale=10))

# "Leakage margin penalty" annotation -- points at the closest-to-boundary triangle
dists = [((px - CB_X) ** 2 + (py - CB_Y) ** 2) ** 0.5 for px, py, _ in lk_after]
i_closest = int(np.argmin(dists))
px, py, _ = lk_after[i_closest]
ax.annotate('Leakage\nmargin penalty', xy=(px, py),
             xytext=(CB_X + 2.35, CB_Y - 2.15),
             fontsize=SMALL_FS, color=C_NEG_LINE, fontweight='bold', ha='center',
             linespacing=1.3,
             arrowprops=dict(arrowstyle='-|>', color=C_NEG_LINE, lw=1.2,
                              connectionstyle='arc3,rad=0.15',
                              mutation_scale=10))

ax.text(PB_X + PB_W / 2, PB_Y + 0.20,
         'center $\\mathbf{c}$ fixed — only the sample geometry changes',
         ha='center', va='bottom', fontsize=TINY_FS, color=C_MUTED, style='italic')

# ═════════════════════════════════════════════════════════════════════════
# COMPACT FORMULA ANNOTATION (shared, between panels and legend)
# ═════════════════════════════════════════════════════════════════════════
FX, FY, FW, FH = 5.25, 1.05, 5.0, 0.80
fbox = FancyBboxPatch(
    (FX, FY), FW, FH, boxstyle='round,pad=0.02,rounding_size=0.16',
    linewidth=1.5, edgecolor=C_FORMULA_EDGE, facecolor=C_FORMULA_FILL, zorder=3,
)
ax.add_patch(fbox)
fcx = FX + FW / 2
ax.text(fcx, FY + FH * 0.68,
         r'$d = \Vert \mathbf{z}^{\mathrm{proj}} - \mathbf{c} \Vert \qquad$'
         r'NoFlow: $d^2 \to 0 \qquad$ Leakage: $d^2 \geq m^2$',
         ha='center', va='center', fontsize=12.0, color=C_TEXT, zorder=4)
ax.text(fcx, FY + FH * 0.24,
         'NoFlow pulled to the center; Leakage penalized only while inside the margin',
         ha='center', va='center', fontsize=TINY_FS, color=C_SUB, zorder=4)

# ═════════════════════════════════════════════════════════════════════════
# SHARED LEGEND
# ═════════════════════════════════════════════════════════════════════════
LEG_Y = 0.45

def legend_swatch_square(x, y, color, sz=0.20):
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz, facecolor=color,
                             edgecolor=C_MARKER_EDGE, linewidth=0.8, zorder=6))

def legend_text(x, y, text, color=C_TEXT, fs=SMALL_FS, weight='normal'):
    ax.text(x, y, text, ha='left', va='center', fontsize=fs, color=color,
             fontweight=weight)

x = 0.5
legend_text(x, LEG_Y, 'Series (color):', fs=SMALL_FS, weight='bold')
x += 1.55
for sname in SERIES_LIST:
    legend_swatch_square(x, LEG_Y, SERIES_COLORS[sname])
    legend_text(x + 0.20, LEG_Y, sname, fs=SMALL_FS)
    x += 0.62

x += 0.45
legend_text(x, LEG_Y, 'Class (shape):', fs=SMALL_FS, weight='bold')
x += 1.65
ax.scatter([x], [LEG_Y], s=140, marker='o', facecolor='#dcdcdc',
           edgecolor=C_MARKER_EDGE, linewidth=0.9, zorder=6)
legend_text(x + 0.28, LEG_Y, 'NoFlow', fs=SMALL_FS)
x += 1.45
ax.scatter([x], [LEG_Y], s=140, marker='^', facecolor='#dcdcdc',
           edgecolor=C_MARKER_EDGE, linewidth=0.9, zorder=6)
legend_text(x + 0.28, LEG_Y, 'Leakage', fs=SMALL_FS)

x += 1.55
ax.scatter([x + 0.10], [LEG_Y], s=260, marker='*', facecolor=C_CENTER,
           edgecolor='#7a5c0a', linewidth=0.8, zorder=6)
legend_text(x + 0.42, LEG_Y, 'center $c$', fs=SMALL_FS)

x += 1.85
ax.add_line(Line2D([x, x + 0.55], [LEG_Y, LEG_Y], color=C_MARGIN, lw=1.6,
                     linestyle=(0, (4, 2.5))))
legend_text(x + 0.72, LEG_Y, 'margin $m$', fs=SMALL_FS)

# ─── save ───────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 600)]:
    p = out_dir / f'deepsad_concept.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
