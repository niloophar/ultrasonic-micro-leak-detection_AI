#!/usr/bin/env python
"""
Thesis figure -- conceptual illustration of latent-space GMM anomaly
scoring and the three calibration protocols used in this thesis.

REBUILT FROM SCRATCH (2026-08-13) to fix three methodological problems
in the previous version:

  1. The old panel (c) right side was titled "Site-wise calibrated GMM"
     but drew the FAIR chronological protocol (earliest-NoFlow-calibrates,
     later-disjoint-data-evaluates) while ALSO carrying a warning box
     reading "historical fit-and-score on the same target NoFlow =
     optimistic upper bound" -- a description that belongs to the
     OPTIMISTIC protocol, not the fair one. Optimistic and fair were
     conflated into a single ambiguous panel.
  2. Only two of the thesis's three protocols were shown (Global +
     one ambiguous site-wise panel). Optimistic site-wise calibration
     was entirely missing.
  3. Panel (a) drew three separate Gaussian components (mu_1, mu_2, mu_3)
     with the caption "modelled as multiple compact modes, not one single
     Gaussian" -- but the actual deployed GMM
     (configs/contrastive/encoder.json: gmm_n_components=1,
     gmm_covariance_type="full"; thesis Sec. "Deployment Feasibility",
     "one mixture component with a full covariance matrix in the
     128-dimensional latent space") has exactly ONE component. The old
     figure visually implied a multi-modal final model that was never
     used.

This version shows:
  - one top panel: patch-level scoring against a SINGLE NoFlow density
    region (matches the deployed K=1 full-covariance GMM), with the
    negative-log-likelihood score and an explicit "not a probability"
    note.
  - three side-by-side bottom panels, one per protocol actually defined
    in Sec. "Global, Optimistic Site-Wise, and Fair Calibrated
    Evaluation" / Table~tab:gmm_evaluation_protocols:
      A. Global GMM -- fit once on the training pool, no recalibration.
      B. Optimistic site-wise -- fit AND scored on the same target
         NoFlow (calibration/evaluation overlap made explicit).
      C. Fair chronological site-wise -- fit on earliest target NoFlow,
         scored only on later, disjoint NoFlow + all Leakage.

Outputs:
  figures/gmm_scoring_concept.pdf
  figures/gmm_scoring_concept.png (600 dpi)

Run: python scripts/plot_thesis_gmm_scoring_concept.py
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle, Arc
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

RNG = np.random.default_rng(7)

# ─── palette ──────────────────────────────────────────────────────────────
C_TEXT = '#232323'
C_SUB = '#5c5c5c'
C_MUTED = '#8f8f8f'
C_MARKER_EDGE = '#333333'
C_GOLD = '#b8860b'

C_NF_FILL = '#dbe8f5'   # single NoFlow density region
C_NF_DARK = '#3f6fa8'

C_POS = '#3f8f7a'   # low score / NoFlow semantics / fair-protocol accent
C_NEG = '#c1663c'   # high score / Leakage semantics
C_OPT = '#b8860b'   # optimistic-protocol accent (amber, matches C_GOLD)

C_PANEL_EDGE = '#9a9a9a'
C_PANEL_FILL = '#fbfbfa'
C_FORMULA_FILL = '#fff7e2'
C_FORMULA_EDGE = '#b8903f'

C_LOCK_EDGE = '#5f6f85'
C_NEUTRAL_FILL = '#c9ccd1'

C_WARN_EDGE = '#b5583f'
C_WARN_FILL = '#fdf1ea'
C_WARN_TEXT = '#8a4226'

C_OK_EDGE = '#3f8f7a'
C_OK_FILL = '#eaf5f1'
C_OK_TEXT = '#255c4a'

LW_PANEL = 1.3

# ─── print-scale font sizing ────────────────────────────────────────────────
# Figure is placed at width=0.95\linewidth on A4 with this thesis's
# 3.0/2.5cm margins -> linewidth = 21 - 3.0 - 2.5 = 15.5cm = 6.10in,
# print width = 0.95 * 6.10 = 5.80in. FIG_W below is in "inches" (1 data
# unit = 1 inch, since figsize=(FIG_W,FIG_H) and xlim/ylim match exactly).
# pt(x) returns the native matplotlib fontsize that prints at x points.
FIG_W, FIG_H = 16.4, 13.6
PRINT_W_IN = 5.80
SCALE = PRINT_W_IN / FIG_W

# Native pt sizes, hand-tuned against this figure's actual box widths
# (panels are ~5in wide for the three-column bottom row) -- NOT derived
# from a print-target formula, which overshot badly on the first pass.
TITLE_FS = 26.0     # top-level headers, full figure width
SUBHEAD_FS = 18.0   # "A. Global GMM" style panel headers
LABEL_FS = 19.0     # formula text, chip captions
SMALL_FS = 18.0     # gauge labels, pill / verdict text
NOTE_FS = 15.0      # short italic notes
LEG_FS = 12.0       # legend row (kept slightly smaller + muted = "lighter")

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')


# ─── helpers ─────────────────────────────────────────────────────────────
def blend(c1, c2, t):
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    return (r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def draw_panel_box(x, y, w, h, fill=C_PANEL_FILL, edge=C_PANEL_EDGE, lw=LW_PANEL):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.16',
                          linewidth=lw, edgecolor=edge, facecolor=fill,
                          linestyle=(0, (5, 3)), zorder=0)
    ax.add_patch(box)


def draw_marker(x, y, shape, color, size=150, zorder=5, edge=C_MARKER_EDGE, lw=1.0):
    m = 'o' if shape == 'circle' else '^'
    ax.scatter([x], [y], s=size, marker=m, facecolor=color, edgecolor=edge,
               linewidth=lw, zorder=zorder)


def draw_query_ring(x, y, r=0.15, zorder=6):
    ring = Circle((x, y), r, fill=False, edgecolor=C_GOLD, linewidth=1.8,
                  linestyle=(0, (3, 1.5)), zorder=zorder)
    ax.add_patch(ring)


def draw_arrow(p_from, p_to, color=C_TEXT, dashed=False, lw=1.3, mscale=9,
               zorder=3, curve=0.0):
    arr = FancyArrowPatch(p_from, p_to, arrowstyle='-|>', mutation_scale=mscale,
                           linewidth=lw, color=color,
                           linestyle=(0, (4, 2.5)) if dashed else '-',
                           connectionstyle=f'arc3,rad={curve}', zorder=zorder,
                           shrinkA=3, shrinkB=3)
    ax.add_patch(arr)


def draw_lock(cx, cy, s, color=C_LOCK_EDGE, zorder=7):
    arc = Arc((cx, cy + s * 0.42), s * 0.62, s * 0.68, angle=0, theta1=15, theta2=165,
              linewidth=1.3, color=color, zorder=zorder)
    ax.add_patch(arc)
    body = FancyBboxPatch((cx - s * 0.38, cy - s * 0.28), s * 0.76, s * 0.56,
                           boxstyle='round,pad=0.0,rounding_size=0.02',
                           linewidth=0.8, edgecolor=color, facecolor=color, zorder=zorder)
    ax.add_patch(body)


def draw_gaussian_blob(cx, cy, rx, ry, angle, fill, n_rings=4, zorder=1):
    scales = [1.35, 1.0, 0.62, 0.32][:n_rings]
    alphas = [0.09, 0.17, 0.30, 0.50][:n_rings]
    for scale, alpha in zip(scales, alphas):
        e = Ellipse((cx, cy), width=2 * rx * scale, height=2 * ry * scale,
                    angle=angle, facecolor=fill, edgecolor='none', alpha=alpha, zorder=zorder)
        ax.add_patch(e)


def draw_chip(cx, cy, w, h, items, caption, cap_color=C_TEXT, fill='#f5f5f3',
              edge='#b8b8b4', cols=3):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle='round,pad=0.02,rounding_size=0.09',
                          linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    for i, (shape, color) in enumerate(items):
        row, col = divmod(i, cols)
        n_rows = -(-len(items) // cols)
        jx = cx + (col - (cols - 1) / 2) * (w * 0.26) + RNG.uniform(-0.02, 0.02)
        jy = cy + (h * 0.22) - row * (h * 0.34) + RNG.uniform(-0.02, 0.02)
        draw_marker(jx, jy, shape, color, size=42, zorder=4, lw=0.7)
    ax.text(cx, cy - h / 2 - 0.10, caption, ha='center', va='top',
            fontsize=NOTE_FS, color=cap_color, linespacing=1.25)


def pill(cx, cy, w, h, text, edge, fill, tcolor, fs=SMALL_FS):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle='round,pad=0.02,rounding_size=0.5',
                          linewidth=1.2, edgecolor=edge, facecolor=fill, zorder=5)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=tcolor,
            fontweight='bold', linespacing=1.2, zorder=6)


# ═══════════════════════════════════════════════════════════════════════
# TOP PANEL — patch-level anomaly scoring (single NoFlow density)
# ═══════════════════════════════════════════════════════════════════════
T_X, T_Y, T_W, T_H = 0.35, 9.75, FIG_W - 0.70, 3.55
draw_panel_box(T_X, T_Y, T_W, T_H)
ax.text(FIG_W / 2, T_Y + T_H + 0.18, 'Patch-level anomaly scoring',
        ha='center', va='bottom', fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

# -- left third: single-component NoFlow density with two query points --
LX = T_X + T_W * 0.155
LY = T_Y + T_H * 0.56
draw_gaussian_blob(LX, LY, 1.55, 1.05, 12, C_NF_FILL)
for _ in range(10):
    u = RNG.normal(0, 0.42) * 1.55
    v = RNG.normal(0, 0.42) * 1.05
    ang = np.deg2rad(12)
    px = LX + u * np.cos(ang) - v * np.sin(ang)
    py = LY + u * np.sin(ang) + v * np.cos(ang)
    draw_marker(px, py, 'circle', C_NF_DARK, size=55, zorder=4, lw=0.7)
ax.scatter([LX], [LY], marker='+', s=170, color=C_NF_DARK, linewidth=2.0, zorder=6)

nf_q = (LX + 0.45, LY + 0.55)
draw_marker(*nf_q, 'circle', C_NF_DARK, size=150, zorder=7)
draw_query_ring(*nf_q, r=0.20)
lk_q = (LX + 1.95, LY - 0.75)
draw_marker(*lk_q, 'triangle', '#8a5a3a', size=150, zorder=7)
draw_query_ring(*lk_q, r=0.20)

ax.text(LX, T_Y + 0.22, 'GMM on NoFlow latents\n(1 component, full covariance)',
        ha='center', va='bottom', fontsize=NOTE_FS, color=C_MUTED, style='italic',
        linespacing=1.3)

# -- middle third: formula --
FX = T_X + T_W * 0.535
fbw, fbh = 5.3, 1.25
fbox = FancyBboxPatch((FX - fbw / 2, LY - fbh / 2), fbw, fbh,
                       boxstyle='round,pad=0.02,rounding_size=0.14',
                       linewidth=1.5, edgecolor=C_FORMULA_EDGE, facecolor=C_FORMULA_FILL, zorder=3)
ax.add_patch(fbox)
ax.text(FX, LY + 0.24, r'$s(\mathbf{x}) = -\log\, p(\mathbf{z}\mid\mathrm{NoFlow})$',
        ha='center', va='center', fontsize=LABEL_FS, color=C_TEXT, zorder=4)
ax.text(FX, LY - 0.32, 'not a leakage probability',
        ha='center', va='center', fontsize=NOTE_FS, color=C_WARN_TEXT, style='italic',
        fontweight='bold', zorder=4)

draw_arrow((LX + 1.55, LY + 0.15), (FX - fbw / 2 - 0.10, LY + 0.06), color=C_NF_DARK, curve=-0.10)
draw_arrow((LX + 2.30, LY - 0.65), (FX - fbw / 2 - 0.10, LY - 0.06), color='#8a5a3a', curve=0.10)

# -- right third: score gauge --
GX0, GX1 = FX + fbw / 2 + 1.0, T_X + T_W - 0.55
GY = LY + 0.05
draw_arrow((FX + fbw / 2 + 0.10, LY), (GX0 - 0.15, GY), lw=1.3)

n_grad = 80
xs = np.linspace(GX0, GX1, n_grad)
for i in range(n_grad - 1):
    tcol = i / (n_grad - 1)
    col = blend(C_POS, C_NEG, tcol)
    ax.add_patch(Rectangle((xs[i], GY - 0.16), xs[i + 1] - xs[i], 0.32,
                            facecolor=col, edgecolor='none', zorder=2))
ax.add_patch(Rectangle((GX0, GY - 0.16), GX1 - GX0, 0.32, fill=False,
                        edgecolor=C_TEXT, linewidth=0.9, zorder=3))
gamma_x = GX0 + 0.55 * (GX1 - GX0)
ax.add_line(Line2D([gamma_x, gamma_x], [GY - 0.28, GY + 0.28], color=C_TEXT, lw=1.5, zorder=4))
ax.text(gamma_x, GY + 0.42, r'$\gamma$', ha='center', va='bottom', fontsize=LABEL_FS,
        color=C_TEXT, fontweight='bold')

nf_score_x = GX0 + 0.20 * (GX1 - GX0)
lk_score_x = GX0 + 0.85 * (GX1 - GX0)
draw_marker(nf_score_x, GY, 'circle', 'white', size=95, zorder=5, edge=C_TEXT, lw=1.3)
draw_marker(lk_score_x, GY, 'triangle', 'white', size=95, zorder=5, edge=C_TEXT, lw=1.3)

ax.text(GX0, GY - 0.55, 'NoFlow', ha='left', va='top',
        fontsize=SMALL_FS, color=C_POS, fontweight='bold')
ax.text(GX1, GY - 0.55, 'Leakage', ha='right', va='top',
        fontsize=SMALL_FS, color=C_NEG, fontweight='bold')
ax.text((GX0 + GX1) / 2, GY - 1.05, 'higher score = lower NoFlow density',
        ha='center', va='top', fontsize=NOTE_FS, color=C_SUB, style='italic')

# ═══════════════════════════════════════════════════════════════════════
# BOTTOM — three calibration protocols, side by side
# ═══════════════════════════════════════════════════════════════════════
SEC_Y = T_Y - 0.35
ax.text(FIG_W / 2, SEC_Y, 'Calibration protocols', ha='center', va='top',
        fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')

B_TOP = SEC_Y - 0.55
B_BOT = 0.75
B_H = B_TOP - B_BOT
gap = 0.35
panel_w = (FIG_W - 0.70 - 2 * gap) / 3
xs0 = [T_X + i * (panel_w + gap) for i in range(3)]

titles = ['A.  Global GMM', 'B.  Optimistic site-wise', 'C.  Fair chronological']
for x0, title in zip(xs0, titles):
    draw_panel_box(x0, B_BOT, panel_w, B_H)
    ax.text(x0 + panel_w / 2, B_BOT + B_H - 0.24, title, ha='center', va='top',
            fontsize=SUBHEAD_FS, color=C_TEXT, fontweight='bold')

chip_w, chip_h = panel_w * 0.40, 1.35

# ── A. Global GMM ──────────────────────────────────────────────────────
ax0 = xs0[0]
cy_a = B_BOT + B_H * 0.60
xa1 = ax0 + panel_w * 0.24
train_items = [('circle', C_NF_DARK)] * 4 + [('circle', '#8f9aa8')] * 2
draw_chip(xa1, cy_a, chip_w, chip_h, train_items, 'training pool\nNoFlow', cols=3)

xa2 = ax0 + panel_w * 0.565
draw_arrow((xa1 + chip_w / 2 + 0.06, cy_a), (xa2 - 0.32, cy_a), lw=1.1)
draw_lock(xa2, cy_a, 0.34)
ax.text(xa2, cy_a - 0.34, 'frozen', ha='center', va='top', fontsize=NOTE_FS,
        color=C_TEXT, fontweight='bold')

xa3 = ax0 + panel_w * 0.855
draw_arrow((xa2 + 0.32, cy_a), (xa3 - chip_w / 2 - 0.06, cy_a), lw=1.1, color=C_NEG)
test_items = [('circle', C_NEUTRAL_FILL), ('triangle', C_NEUTRAL_FILL)] * 3
draw_chip(xa3, cy_a, chip_w, chip_h, test_items, 'target series\n(scored)', cols=2)

pill(ax0 + panel_w / 2, B_BOT + 0.42, panel_w - 0.5, 0.55, 'no target recalibration',
     C_NEG, '#fbeee6', C_NEG, fs=NOTE_FS)

# ── B. Optimistic site-wise ─────────────────────────────────────────────
bx0 = xs0[1]
cy_b = B_BOT + B_H * 0.61
xb_mid = bx0 + panel_w / 2
tgt_items = [('circle', C_OPT)] * 6
draw_chip(xb_mid, cy_b, chip_w * 1.15, chip_h, tgt_items, 'ALL target-site NoFlow', cols=3)

# single-line caption sits right below the chip -- start the fan-out arrows
# well below it (not at the chip edge) with generous clearance throughout.
cap_clear_b = cy_b - chip_h / 2 - 0.55
y_fit = cap_clear_b - 0.65
y_eval = y_fit
xb_left = bx0 + panel_w * 0.16
xb_right = bx0 + panel_w * 0.84
draw_arrow((xb_mid - 0.35, cap_clear_b), (xb_left, y_fit + 0.32), lw=1.1, color=C_OPT, curve=-0.08)
draw_arrow((xb_mid + 0.35, cap_clear_b), (xb_right, y_eval + 0.32), lw=1.1, color=C_OPT, curve=0.08)

ax.text(xb_left, y_fit, 'fit GMM +\nq95 threshold', ha='center', va='top', fontsize=NOTE_FS,
        color=C_OPT, fontweight='bold', linespacing=1.25)
ax.text(xb_right, y_eval, 'score same NoFlow\n+ Leakage', ha='center', va='top',
        fontsize=NOTE_FS, color=C_OPT, fontweight='bold', linespacing=1.25)

# overlap loop between the two arrows' targets -- makes "same data" explicit
loop_y = y_fit - 0.90
ax.add_line(Line2D([xb_left, xb_right], [loop_y, loop_y], color=C_OPT, lw=1.2,
                    linestyle=(0, (3, 2)), zorder=2))
draw_arrow((xb_left, loop_y), (xb_left, loop_y + 0.02), color=C_OPT, lw=1.2, mscale=7)
ax.text(xb_mid, loop_y - 0.18, 'calibration and evaluation NoFlow overlap',
        ha='center', va='top', fontsize=NOTE_FS, color=C_WARN_TEXT, style='italic')

pill(bx0 + panel_w / 2, B_BOT + 0.32, panel_w - 0.5, 0.55, 'optimistic upper bound',
     C_WARN_EDGE, C_WARN_FILL, C_WARN_TEXT, fs=NOTE_FS)

# ── C. Fair chronological ───────────────────────────────────────────────
cx0 = xs0[2]
tl_y = B_BOT + B_H * 0.76
tl_x0, tl_x1 = cx0 + 0.45, cx0 + panel_w - 0.45
draw_arrow((tl_x0, tl_y), (tl_x1, tl_y), lw=1.0, color=C_MUTED, mscale=8)
ax.text(tl_x1, tl_y + 0.14, 'time', ha='right', va='bottom', fontsize=NOTE_FS,
        color=C_MUTED, style='italic')

calib_cx = tl_x0 + (tl_x1 - tl_x0) * 0.19
test_cx = tl_x0 + (tl_x1 - tl_x0) * 0.81
divider_x = tl_x0 + (tl_x1 - tl_x0) * 0.50
cy_c = tl_y - 1.15
ax.add_line(Line2D([divider_x, divider_x], [tl_y - 0.15, cy_c - chip_h / 2 + 0.08],
                    color=C_MUTED, lw=1.0, linestyle=(0, (2, 2)), zorder=2))
ax.text(divider_x, tl_y + 0.14, 'disjoint', ha='center', va='bottom', fontsize=NOTE_FS,
        color=C_MUTED, style='italic')

calib_items = [('circle', C_POS)] * 6
draw_chip(calib_cx, cy_c, chip_w, chip_h, calib_items, 'earliest NoFlow\n(calibration)', cols=3)
test3_items = [('circle', C_NEUTRAL_FILL), ('triangle', '#8a5a3a')] * 3
draw_chip(test_cx, cy_c, chip_w, chip_h, test3_items, 'later NoFlow +\nLeakage', cols=2)

# chip captions are 2 lines (~0.55in tall below the chip edge) -- arrows
# start well below them, not at the chip edge, for clear visual separation.
cap_clear = cy_c - chip_h / 2 - 0.80
y_c2 = cap_clear - 0.40
draw_arrow((calib_cx, cap_clear), (calib_cx, y_c2 + 0.32), lw=1.1, color=C_POS)
ax.text(calib_cx, y_c2, 'fit GMM +\nq95 threshold', ha='center', va='top', fontsize=NOTE_FS,
        color=C_POS, fontweight='bold', linespacing=1.25)
draw_arrow((test_cx, cap_clear), (test_cx, y_c2 + 0.32), lw=1.1, color=C_NEG)
ax.text(test_cx, y_c2, 'evaluate\n(held out)', ha='center', va='top', fontsize=NOTE_FS,
        color=C_NEG, fontweight='bold', linespacing=1.25)

pill(cx0 + panel_w / 2, B_BOT + 0.32, panel_w - 0.5, 0.55, 'calib. / eval. disjoint',
     C_OK_EDGE, C_OK_FILL, C_OK_TEXT, fs=NOTE_FS)

# ═══════════════════════════════════════════════════════════════════════
# SHARED LEGEND
# ═══════════════════════════════════════════════════════════════════════
LEG_Y = 0.24
x = 0.55
ax.scatter([x], [LEG_Y], s=85, marker='o', facecolor='#dcdcdc', edgecolor=C_MARKER_EDGE,
           linewidth=0.8, zorder=6)
ax.text(x + 0.20, LEG_Y, 'NoFlow', ha='left', va='center', fontsize=LEG_FS, color=C_SUB)
x += 1.35
ax.scatter([x], [LEG_Y], s=85, marker='^', facecolor='#dcdcdc', edgecolor=C_MARKER_EDGE,
           linewidth=0.8, zorder=6)
ax.text(x + 0.20, LEG_Y, 'Leakage', ha='left', va='center', fontsize=LEG_FS, color=C_SUB)
x += 1.55
ax.add_patch(Ellipse((x + 0.14, LEG_Y), width=0.42, height=0.26, facecolor=C_NF_FILL,
                      edgecolor=C_NF_DARK, linewidth=0.9, alpha=0.9, zorder=6))
ax.text(x + 0.48, LEG_Y, 'NoFlow density', ha='left', va='center',
        fontsize=LEG_FS, color=C_SUB)
x += 2.40
ax.scatter([x + 0.07], [LEG_Y], marker='+', s=120, color=C_NF_DARK, linewidth=1.7, zorder=6)
ax.text(x + 0.24, LEG_Y, r'mean $\mu$', ha='left', va='center', fontsize=LEG_FS, color=C_SUB)
x += 1.50
ring = Circle((x + 0.09, LEG_Y), 0.11, fill=False, edgecolor=C_GOLD, linewidth=1.4,
              linestyle=(0, (3, 1.5)))
ax.add_patch(ring)
ax.text(x + 0.28, LEG_Y, 'query patch', ha='left', va='center', fontsize=LEG_FS, color=C_SUB)
x += 1.85
draw_lock(x + 0.09, LEG_Y, 0.20)
ax.text(x + 0.28, LEG_Y, 'frozen GMM', ha='left', va='center', fontsize=LEG_FS, color=C_SUB)

# ─── save ───────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 600)]:
    p = out_dir / f'gmm_scoring_concept.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved -> {p}')

plt.close(fig)

print()
print('=== gmm_scoring_concept -- verified against local evidence ===')
print('  GMM final config (configs/contrastive/encoder.json):')
print('    gmm_n_components = 1, gmm_covariance_type = "full", latent_dim = 128')
print('    -> panel now shows ONE NoFlow density region, not a multi-modal mixture.')
print('  Three protocols (thesis Sec. "Global, Optimistic Site-Wise, and Fair')
print('  Calibrated Evaluation", Table tab:gmm_evaluation_protocols):')
print('    A. Global        -- fit on training pool, no recalibration')
print('    B. Optimistic    -- fit + scored on the SAME target NoFlow (overlap)')
print('    C. Fair chrono.  -- fit on earliest target NoFlow, scored on later,')
print('                        disjoint NoFlow + all Leakage')
