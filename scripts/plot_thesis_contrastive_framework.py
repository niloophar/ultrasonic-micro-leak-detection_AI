#!/usr/bin/env python
"""
Thesis figure -- integrated architecture of the contrastive representation
learning + latent-space anomaly detection framework.

Paper-style schematic, four dashed grouped regions:
  1. Input Representation    -- G5 patch tensor x in R^(2x9x64), shown as a
                                 small layered heatmap icon.
  2. Representation Learning -- backbone encoder f_theta (stacked-block icon)
                                 -> latent z -> decoder / projection branches.
  3. Latent-Space Objectives -- SupCon / DeepSAD (RNC de-emphasized) +
                                 compact total-loss formula.
  4. Deployment / Anomaly Scoring -- frozen encoder -> latent cloud + GMM
                                 density metaphor -> score -> threshold ->
                                 decision -> majority vote.

A single dashed rail in the empty left-middle column carries the
"freeze encoder, discard decoder & projection head" transition from
training to deployment.

Outputs:
  figures/contrastive_framework.pdf
  figures/contrastive_framework.png (600 dpi)

Run: python scripts/plot_thesis_contrastive_framework.py
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Arc, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

# ─── palette ────────────────────────────────────────────────────────────────
C_TEXT = '#232323'
C_SUB = '#5c5c5c'
C_MUTED = '#8f8f8f'

# panel identity colors (blue / teal / lavender / warm amber)
C_P1_EDGE, C_P1_FILL = '#5b8fc2', '#f5f8fc'
C_P2_EDGE, C_P2_FILL = '#4f9a8f', '#f3f9f8'
C_P3_EDGE, C_P3_FILL = '#8a7ec4', '#f6f4fb'
C_P4_EDGE, C_P4_FILL = '#c98a3d', '#fdf8f0'

# component colors
C_ENC_LIGHT, C_ENC_DARK, C_ENC_EDGE = '#dcefe9', '#79b7ab', '#3f7f74'
C_DEC_LIGHT, C_DEC_DARK, C_DEC_EDGE = '#eef1ef', '#c3cdc9', '#8a9793'
C_PROJ_FILL, C_PROJ_EDGE = '#e7e0f3', '#8a7ec4'
C_LATENT_DOT = '#4f9a8f'
C_FORMULA_FILL, C_FORMULA_EDGE = '#fff7e2', '#b8903f'
C_SUPCON_FILL, C_SUPCON_EDGE = '#e4f3f0', '#4f9a8f'
C_DEEPSAD_FILL, C_DEEPSAD_EDGE = '#e9f0fa', '#5b8fc2'
C_RNC_FILL, C_RNC_EDGE, C_RNC_TEXT = '#ececec', '#a8a8a8', '#8f8f8f'
C_PULL = '#4f9a8f'
C_PUSH = '#d97a4a'

C_FROZEN_LIGHT, C_FROZEN_DARK, C_FROZEN_EDGE = '#dde4ee', '#a9b6c9', '#5f6f85'
C_NOFLOW_DOT = '#6f9fd0'
C_GMM_LINE = '#5b8fc2'
C_QUERY = '#d9683f'
C_SCORE_FILL, C_SCORE_EDGE = '#fdf1de', '#c98a3d'
C_THRESH_GREEN, C_THRESH_RED = '#93c1a0', '#e0906c'
C_DECISION_GREEN, C_DECISION_RED = '#e1f1e4', '#fbe6dc'
C_MV_FILL, C_MV_EDGE = '#eef0f4', '#9aa3b0'
C_MV_HILITE = '#4f9a8f'

C_ANNOT_EDGE = '#9a9a9a'
C_ARROW = '#4a4a4a'
C_ARROW_LIGHT = '#a3a3a3'

LW_PANEL = 1.25
LW_BOX = 1.05
LW_ARR = 1.05

TITLE_FS = 8.6
SUB_FS = 7.6
PANEL_LABEL_FS = 6.6
SMALL_FS = 6.6
TINY_FS = 6.0

# ─── figure setup ────────────────────────────────────────────────────────────
FIG_W, FIG_H = 17.0, 11.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Print-legibility fix: this figure is designed at FIG_W=17.0in but placed
# in the thesis via a landscape/sideways figure (~9.5in usable page height
# after rotation, see the regeneration audit) rather than a portrait
# \linewidth column -- box positions below are all absolute, hand-tuned
# coordinates, so growing individual boxes (as done for the feature-group
# figure) isn't a safe option here without redoing the whole layout.
# Instead every ax.text() fontsize is scaled uniformly at the call site via
# this wrapper, catching both the named *_FS constants and the many inline
# literal font sizes below in one place -- no box, arrow, icon, or
# coordinate is touched, so the diagram's content/positions are unchanged.
FONT_MULT = 2.35
_orig_ax_text = ax.text
def _scaled_ax_text(*args, **kwargs):
    if 'fontsize' in kwargs:
        kwargs['fontsize'] = kwargs['fontsize'] * FONT_MULT
    return _orig_ax_text(*args, **kwargs)
ax.text = _scaled_ax_text


# ─── generic helpers ─────────────────────────────────────────────────────────
def blend(c1, c2, t):
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    return (r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def draw_panel(x, y, w, h, label, edge, fill):
    panel = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.24',
        linewidth=LW_PANEL, edgecolor=edge, facecolor=fill,
        linestyle=(0, (5, 3)), zorder=0,
    )
    ax.add_patch(panel)
    ax.text(x + 0.24, y + h - 0.16, label, ha='left', va='top',
             fontsize=PANEL_LABEL_FS, color=edge, fontweight='bold', zorder=1)


def draw_box(x, y, w, h, title, subtitle=None, fill='#f4f4f5', edge=C_TEXT,
             dashed=False, lw=LW_BOX, title_fs=TITLE_FS, sub_fs=SUB_FS,
             title_color=C_TEXT, sub_color=C_SUB, rounding=0.09, zorder=3,
             title_weight='bold'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.02,rounding_size={rounding}',
        linewidth=lw, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 2)) if dashed else '-', zorder=zorder,
    )
    ax.add_patch(box)
    cx = x + w / 2
    if subtitle is None:
        ax.text(cx, y + h / 2, title, ha='center', va='center',
                 fontsize=title_fs, color=title_color, fontweight=title_weight,
                 zorder=zorder + 1)
    else:
        ax.text(cx, y + h * 0.66, title, ha='center', va='center',
                 fontsize=title_fs, color=title_color, fontweight=title_weight,
                 zorder=zorder + 1)
        ax.text(cx, y + h * 0.28, subtitle, ha='center', va='center',
                 fontsize=sub_fs, color=sub_color, zorder=zorder + 1,
                 linespacing=1.3)
    return x, y, w, h


def draw_arrow(p_from, p_to, color=C_ARROW, dashed=False, lw=LW_ARR,
               mscale=9, zorder=2, connectionstyle=None):
    arr = FancyArrowPatch(
        p_from, p_to,
        arrowstyle='-|>', mutation_scale=mscale,
        linewidth=lw, color=color,
        linestyle=(0, (4, 2.5)) if dashed else '-',
        connectionstyle=connectionstyle,
        zorder=zorder, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)


def draw_link(p_from, p_to, color=C_ANNOT_EDGE, lw=0.8, zorder=1):
    ax.add_line(Line2D([p_from[0], p_to[0]], [p_from[1], p_to[1]],
                        color=color, lw=lw, linestyle=(0, (1, 1.6)), zorder=zorder))


# ─── schematic icons ─────────────────────────────────────────────────────────
def draw_stack(x0, y0, w, h, n=4, offset=0.085, light='#dfeee9', dark='#79b7ab',
                edge='#3f7f74', lw=1.0, zorder=3):
    """Layered rounded blocks (back -> front), bottom-left of the FRONT
    (foreground, largest-offset-index-0) layer anchored at (x0, y0)."""
    for i in range(n - 1, -1, -1):
        ox, oy = i * offset, i * offset
        t = i / max(n - 1, 1)
        fill = blend(light, dark, 1 - t)
        box = FancyBboxPatch(
            (x0 + ox, y0 + oy), w, h,
            boxstyle='round,pad=0.02,rounding_size=0.055',
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=zorder + (n - i),
        )
        ax.add_patch(box)
    bbox = (x0, y0, x0 + w + (n - 1) * offset, y0 + h + (n - 1) * offset)
    return bbox


def draw_tensor_icon(x0, y0, w, h, n_layers=2, offset=0.15,
                      cmaps=('PuBu', 'BuGn'), rows=9, cols=48, seed=0, zorder=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4.5 * np.pi, cols)
    profile = np.linspace(0, 2.2, rows)
    for i in range(n_layers - 1, -1, -1):
        ox, oy = i * offset, i * offset
        img = np.sin(t[None, :] + profile[:, None]) + 0.35 * rng.standard_normal((rows, cols))
        extent = (x0 + ox, x0 + ox + w, y0 + oy, y0 + oy + h)
        ax.imshow(img, extent=extent, aspect='auto', cmap=cmaps[i % len(cmaps)],
                   zorder=zorder + (n_layers - i), interpolation='bilinear')
        rect = Rectangle((extent[0], extent[2]), w, h, linewidth=0.9,
                          edgecolor='#555555', facecolor='none',
                          zorder=zorder + (n_layers - i) + 0.5)
        ax.add_patch(rect)
    bbox = (x0, y0, x0 + w + (n_layers - 1) * offset, y0 + h + (n_layers - 1) * offset)
    return bbox


def draw_cloud(cx, cy, rx, ry, color, n=9, seed=0, s=11, zorder=4):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(0, 1, n))
    xs = cx + rx * rad * np.cos(ang)
    ys = cy + ry * rad * np.sin(ang)
    ax.scatter(xs, ys, s=s, color=color, zorder=zorder, edgecolor='white', linewidth=0.35)
    return xs, ys


def draw_gmm_ellipses(cx, cy, rx, ry, n=3, color=C_GMM_LINE, zorder=3):
    for i in range(n, 0, -1):
        frac = i / n
        e = Ellipse((cx, cy), width=2 * rx * frac, height=2 * ry * frac, fill=False,
                    edgecolor=color, linewidth=1.0, linestyle=(0, (3.5, 2)),
                    alpha=0.30 + 0.30 * (1 - frac), zorder=zorder)
        ax.add_patch(e)


def draw_lock(cx, cy, s, color=C_FROZEN_EDGE, zorder=6):
    arc = Arc((cx, cy + s * 0.42), s * 0.62, s * 0.68, angle=0, theta1=15, theta2=165,
               linewidth=1.2, color=color, zorder=zorder)
    ax.add_patch(arc)
    body = FancyBboxPatch((cx - s * 0.38, cy - s * 0.28), s * 0.76, s * 0.56,
                            boxstyle='round,pad=0.0,rounding_size=0.02',
                            linewidth=0.8, edgecolor=color, facecolor=color, zorder=zorder)
    ax.add_patch(body)


def draw_pull_push(cx, cy, r=0.62):
    circ = Ellipse((cx, cy), width=2 * r, height=2 * r * 0.92, fill=False,
                    edgecolor=C_MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    ax.add_patch(circ)
    rng = np.random.default_rng(7)
    n = 5
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = 0.24 * np.sqrt(rng.uniform(0, 1, n))
    xs = cx + rad * np.cos(ang)
    ys = cy + rad * np.sin(ang) * 0.9
    ax.scatter(xs, ys, s=15, color=C_PULL, zorder=4, edgecolor='white', linewidth=0.3)
    draw_arrow((cx - r * 0.92, cy + 0.05), (cx - 0.28, cy + 0.02), color=C_PULL,
               lw=0.8, mscale=6, zorder=4)
    for dx, dy in [(0.86, 0.30), (0.9, -0.24)]:
        ax.scatter([cx + dx], [cy + dy], s=20, color=C_PUSH, marker='D', zorder=4,
                   edgecolor='white', linewidth=0.3)
        draw_arrow((cx + 0.30, cy - 0.02), (cx + dx - 0.10, cy + dy * 0.7), color=C_PUSH,
                   lw=0.8, mscale=6, dashed=True, zorder=4)
    ax.text(cx, cy - r * 1.28, 'pull · same class', ha='center', va='top',
             fontsize=TINY_FS, color=C_PULL, style='italic')
    ax.text(cx, cy - r * 1.28 - 0.30, 'push · different class', ha='center', va='top',
             fontsize=TINY_FS, color=C_PUSH, style='italic')


def draw_threshold_bar(x0, y0, w, h):
    n = 60
    xs = np.linspace(x0, x0 + w, n)
    for i in range(n - 1):
        t = i / (n - 1)
        col = blend(C_THRESH_GREEN, C_THRESH_RED, t)
        ax.add_patch(Rectangle((xs[i], y0), xs[i + 1] - xs[i], h, facecolor=col,
                                 edgecolor='none', zorder=3))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=C_TEXT, linewidth=0.8, zorder=4))
    thr_x = x0 + 0.6 * w
    ax.add_line(Line2D([thr_x, thr_x], [y0 - 0.06, y0 + h + 0.06], color=C_TEXT, lw=1.2, zorder=5))


def draw_majority_icons(cx, cy, n=5, s=0.24, gap=0.10, hilite=3):
    total_w = n * s + (n - 1) * gap
    x0 = cx - total_w / 2
    for i in range(n):
        x = x0 + i * (s + gap)
        is_h = i < hilite
        fill = blend(C_MV_HILITE, '#ffffff', 0.15) if is_h else '#f4f4f4'
        edge = C_MV_HILITE if is_h else C_MV_EDGE
        box = FancyBboxPatch((x, cy - s / 2), s, s, boxstyle='round,pad=0.0,rounding_size=0.03',
                               linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=4)
        ax.add_patch(box)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 1 — INPUT REPRESENTATION            x:0.35–3.85   y:7.55–11.10
# ═════════════════════════════════════════════════════════════════════════
P1_X, P1_Y, P1_W, P1_H = 0.35, 7.55, 3.50, 3.55
draw_panel(P1_X, P1_Y, P1_W, P1_H, 'Input Representation', C_P1_EDGE, C_P1_FILL)

tb = draw_tensor_icon(1.25, 8.95, 1.55, 1.00, n_layers=2, offset=0.15, seed=1)
ax.text(P1_X + P1_W / 2, 8.62, 'Input patch', ha='center', va='center',
         fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')
ax.text(P1_X + P1_W / 2, 8.24, r'$\mathbf{x}\in\mathbb{R}^{2\times9\times64}$',
         ha='center', va='center', fontsize=12.5, color=C_TEXT)
ax.text(P1_X + P1_W / 2, 7.83, 'G5 · 2 channels · 9 features · 64 steps',
         ha='center', va='center', fontsize=SMALL_FS, color=C_SUB, style='italic')

# ═════════════════════════════════════════════════════════════════════════
# PANEL 2 — REPRESENTATION LEARNING          x:4.15–16.65  y:7.55–11.10
# ═════════════════════════════════════════════════════════════════════════
P2_X, P2_Y, P2_W, P2_H = 4.15, 7.55, 12.50, 3.55
draw_panel(P2_X, P2_Y, P2_W, P2_H, 'Representation Learning', C_P2_EDGE, C_P2_FILL)

draw_arrow((tb[2] + 0.06, 9.45), (4.85, 9.35), lw=1.0, mscale=9)

eb = draw_stack(4.95, 8.75, 1.35, 1.15, n=4, offset=0.075,
                 light=C_ENC_LIGHT, dark=C_ENC_DARK, edge=C_ENC_EDGE)
enc_cx = (eb[0] + eb[2]) / 2
ax.text(enc_cx, 8.55, 'Encoder', ha='center', va='center',
         fontsize=TITLE_FS, color=C_TEXT, fontweight='bold')
ax.text(enc_cx, 8.27, r'$f_{\theta}$  ·  CNN / Transformer', ha='center', va='center',
         fontsize=SUB_FS, color=C_SUB)
ax.text(enc_cx, 7.95, 'series-stratified batches → cross-series invariance',
         ha='center', va='center', fontsize=TINY_FS, color=C_MUTED, style='italic')

# latent z
lat_cx, lat_cy = 8.05, 9.30
draw_arrow((eb[2] + 0.05, 9.30), (lat_cx - 0.42, lat_cy), lw=1.0, mscale=9)
circ = Ellipse((lat_cx, lat_cy), width=0.86, height=0.72, fill=False,
                edgecolor=C_LATENT_DOT, linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
ax.add_patch(circ)
draw_cloud(lat_cx, lat_cy, 0.24, 0.19, C_LATENT_DOT, n=7, seed=4, s=13)
ax.text(lat_cx, lat_cy - 0.62, r'$\mathbf{z}\in\mathbb{R}^{d_z}$', ha='center', va='center',
         fontsize=10.5, color=C_TEXT)

# branch split
draw_arrow((lat_cx + 0.30, lat_cy + 0.22), (9.05, 10.15), lw=0.95, mscale=8)
draw_arrow((lat_cx + 0.30, lat_cy - 0.22), (9.05, 8.35), lw=0.95, mscale=8)

# reconstruction branch (upper)
dbox = draw_box(9.05, 9.95, 1.65, 0.62, 'Decoder', None, fill=blend(C_DEC_LIGHT, C_DEC_DARK, 0.35),
                 edge=C_DEC_EDGE, title_fs=8.0)
draw_arrow((10.70, 10.26), (11.30, 10.26), lw=0.95, mscale=8)
rb = draw_tensor_icon(11.35, 10.00, 0.95, 0.55, n_layers=1, cmaps=('PuBu',), seed=2)
ax.text((rb[0] + rb[2]) / 2, 9.86, r'$\hat{\mathbf{x}}$', ha='center', va='center',
         fontsize=9.5, color=C_TEXT)
draw_arrow((rb[2] + 0.10, 10.26), (12.75, 10.26), lw=0.95, mscale=8)
draw_box(12.78, 9.98, 1.30, 0.56, r'$\mathcal{L}_{\mathrm{rec}}$', None,
         fill=C_FORMULA_FILL, edge=C_FORMULA_EDGE, title_fs=10.5, rounding=0.28)

# projection branch (lower)
pbox = draw_box(9.05, 8.05, 1.65, 0.62, 'Projection', r'$g_{\phi}$', fill=C_PROJ_FILL,
                 edge=C_PROJ_EDGE, title_fs=8.0, sub_fs=8.5)
draw_arrow((10.70, 8.36), (11.30, 8.36), lw=0.95, mscale=8)
zp_cx, zp_cy = 11.75, 8.36
ax.add_patch(Ellipse((zp_cx, zp_cy), width=0.66, height=0.66, fill=False,
                       edgecolor=C_PROJ_EDGE, linewidth=1.0, zorder=3))
ax.scatter([zp_cx + 0.33 * np.cos(0.6)], [zp_cy + 0.33 * np.sin(0.6)], s=18,
           color=C_PROJ_EDGE, zorder=4)
ax.add_line(Line2D([zp_cx, zp_cx + 0.33 * np.cos(0.6)], [zp_cy, zp_cy + 0.33 * np.sin(0.6)],
                    color=C_PROJ_EDGE, lw=0.8, zorder=3))
ax.text(zp_cx, zp_cy - 0.55, r'$\mathbf{z}^{\mathrm{proj}}$', ha='center', va='center',
         fontsize=9.5, color=C_TEXT)

# ═════════════════════════════════════════════════════════════════════════
# PANEL 3 — LATENT-SPACE OBJECTIVES          x:4.15–16.65  y:3.85–7.30
# ═════════════════════════════════════════════════════════════════════════
P3_X, P3_Y, P3_W, P3_H = 4.15, 3.85, 12.50, 3.45
draw_panel(P3_X, P3_Y, P3_W, P3_H, 'Latent-Space Objectives', C_P3_EDGE, C_P3_FILL)

draw_arrow((zp_cx, 8.05), (zp_cx, 6.85), lw=1.0, mscale=9, connectionstyle='arc3,rad=0.0')
draw_arrow((13.42, 9.98), (13.42, 6.85), lw=1.0, mscale=9)

draw_pull_push(6.35, 5.55, r=0.60)

supcon = draw_box(8.30, 6.35, 2.05, 0.62, 'SupCon', r'$\mathcal{L}_{\mathrm{SupCon}}$',
                   fill=C_SUPCON_FILL, edge=C_SUPCON_EDGE, title_fs=8.6, sub_fs=9.0)
deepsad = draw_box(8.30, 5.55, 2.05, 0.62, 'DeepSAD', r'$\mathcal{L}_{\mathrm{DeepSAD}}$',
                    fill=C_DEEPSAD_FILL, edge=C_DEEPSAD_EDGE, title_fs=8.6, sub_fs=9.0)
draw_box(8.30, 4.85, 1.55, 0.44, 'RNC', 'exploratory', fill=C_RNC_FILL, edge=C_RNC_EDGE,
         dashed=True, title_fs=6.9, sub_fs=6.0, title_color=C_RNC_TEXT, sub_color=C_RNC_TEXT)

draw_arrow((7.00, 5.75), (8.25, 6.62), lw=0.85, mscale=7, color=C_MUTED)
draw_arrow((7.00, 5.35), (8.25, 5.82), lw=0.85, mscale=7, color=C_MUTED)

draw_arrow((10.35, 6.66), (11.15, 6.20), lw=0.95, mscale=8)
draw_arrow((10.35, 5.86), (11.15, 5.95), lw=0.95, mscale=8)

fbox = draw_box(11.20, 4.55, 5.05, 1.75, '', fill=C_FORMULA_FILL, edge=C_FORMULA_EDGE, lw=1.4,
                 rounding=0.14)
fcx = 11.20 + 5.05 / 2
ax.text(fcx, 5.75, r'$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{lat}}\,\mathcal{L}_{\mathrm{lat}}$',
         ha='center', va='center', fontsize=13.5, color=C_TEXT, zorder=4)
ax.text(fcx, 5.12, r'$\mathcal{L}_{\mathrm{lat}} \in \{\mathcal{L}_{\mathrm{SupCon}},\ \mathcal{L}_{\mathrm{DeepSAD}}\}$',
         ha='center', va='center', fontsize=9.6, color=C_SUB, zorder=4)

# ─── transition note: freeze encoder / discard decoder + projection head ──
NOTE_X = 2.05
ax.text(NOTE_X, 5.85, 'after training —\nfreeze $f_{\\theta}$,\ndiscard decoder &\nprojection head',
         ha='center', va='center', fontsize=7.3, color=C_SUB, style='italic',
         linespacing=1.5, zorder=4,
         bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=C_ANNOT_EDGE, lw=0.8,
                    linestyle=(0, (3, 2))))
draw_arrow((NOTE_X, 4.75), (1.55, 2.75), color=C_MUTED, dashed=True, lw=1.0, mscale=9,
           connectionstyle='arc3,rad=-0.15')

# ═════════════════════════════════════════════════════════════════════════
# PANEL 4 — DEPLOYMENT / ANOMALY SCORING     x:0.35–16.65  y:0.30–3.60
# ═════════════════════════════════════════════════════════════════════════
P4_X, P4_Y, P4_W, P4_H = 0.35, 0.30, 16.30, 3.30
draw_panel(P4_X, P4_Y, P4_W, P4_H, 'Deployment / Anomaly Scoring', C_P4_EDGE, C_P4_FILL)

fb = draw_stack(0.85, 1.55, 1.05, 0.95, n=3, offset=0.07,
                 light=C_FROZEN_LIGHT, dark=C_FROZEN_DARK, edge=C_FROZEN_EDGE)
fenc_cx = (fb[0] + fb[2]) / 2
draw_lock(fb[2] - 0.04, fb[3] - 0.02, 0.22)
ax.text(fenc_cx, 1.28, 'Frozen encoder', ha='center', va='center',
         fontsize=7.6, color=C_TEXT, fontweight='bold')
ax.text(fenc_cx, 1.03, r'$f_{\theta}$', ha='center', va='center', fontsize=8.5, color=C_SUB)

draw_arrow((fb[2] + 0.08, 1.95), (3.55, 1.95), lw=1.0, mscale=9)

# latent space / GMM metaphor
gmm_cx, gmm_cy = 4.65, 2.05
draw_gmm_ellipses(gmm_cx, gmm_cy, rx=0.95, ry=0.62, n=3)
draw_cloud(gmm_cx, gmm_cy, 0.42, 0.28, C_NOFLOW_DOT, n=14, seed=6, s=11)
qx, qy = gmm_cx + 1.05, gmm_cy + 0.55
ax.scatter([qx], [qy], s=42, color=C_QUERY, marker='D', zorder=6,
           edgecolor='white', linewidth=0.5)
draw_link((gmm_cx, gmm_cy), (qx, qy), color=C_QUERY, lw=0.8)
ax.text(qx + 0.14, qy + 0.03, r'$\mathbf{x}_{\mathrm{query}}$', ha='left', va='center',
         fontsize=7.0, color=C_QUERY)
ax.text(gmm_cx, gmm_cy - 1.02, 'GMM on NoFlow latents', ha='center', va='center',
         fontsize=7.6, color=C_TEXT, fontweight='bold')
ax.text(gmm_cx, gmm_cy - 1.27, 'calibration window', ha='center', va='center',
         fontsize=TINY_FS, color=C_MUTED, style='italic')

draw_arrow((5.95, 2.05), (6.55, 2.05), lw=1.0, mscale=9)
draw_box(6.60, 1.68, 2.55, 0.74, r'$s(\mathbf{x}) = -\log p(\mathbf{z}\mid\mathrm{NoFlow})$',
         None, fill=C_SCORE_FILL, edge=C_SCORE_EDGE, title_fs=9.3, rounding=0.20)
ax.text(6.60 + 2.55 / 2, 1.48, 'anomaly score', ha='center', va='center',
         fontsize=TINY_FS, color=C_MUTED, style='italic')

draw_arrow((9.20, 2.05), (9.75, 2.05), lw=1.0, mscale=9)
draw_threshold_bar(9.80, 1.85, 1.55, 0.38)
ax.text(9.80 + 1.55 / 2, 1.48, 'threshold', ha='center', va='center',
         fontsize=TINY_FS, color=C_MUTED, style='italic')

draw_arrow((11.40, 2.05), (11.90, 2.05), lw=1.0, mscale=9)
draw_box(11.95, 1.68, 2.10, 0.74, '', fill='#fbfbfa', edge=C_TEXT, lw=1.0, rounding=0.30)
ax.add_line(Line2D([13.00, 13.00], [1.78, 2.32], color='#c7c7c7', lw=0.9, zorder=4))
ax.text(12.475, 2.05, 'NoFlow', ha='center', va='center', fontsize=3.7,
         color='#3f7f74', fontweight='bold', zorder=5)
ax.text(13.525, 2.05, 'Leakage', ha='center', va='center', fontsize=3.7,
         color='#c1663c', fontweight='bold', zorder=5)
ax.text(13.00, 1.48, 'patch-level decision', ha='center', va='center',
         fontsize=TINY_FS, color=C_MUTED, style='italic')

draw_arrow((14.05, 2.05), (14.55, 2.05), lw=1.0, mscale=9, dashed=True, color=C_ARROW_LIGHT)
draw_majority_icons(15.50, 2.05, n=5, hilite=4)
ax.text(15.50, 1.60, 'majority vote', ha='center', va='center', fontsize=7.3,
         color=C_TEXT, fontweight='bold')
ax.text(15.50, 1.36, 'file-level decision', ha='center', va='center', fontsize=TINY_FS,
         color=C_MUTED, style='italic')

# ─── save ─────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 600)]:
    p = out_dir / f'contrastive_framework.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
