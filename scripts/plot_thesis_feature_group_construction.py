#!/usr/bin/env python
"""
Thesis figure — schematic construction of feature groups G5/G6/G7/G8 from
one ultrasonic measurement cycle. Methodological diagram, not a data plot.

Every formula shown is read directly from src/preparation/features.py
(compute_features, fit_temperature_correction, apply_temperature_correction)
— none are invented. In particular:

- G7 (echo_ratio_i) is verified to be INTERVAL-based:
      echo_ratio_i = interval_up_i / interval_down_i
  NOT an amplitude ratio. The "echo" in the name refers to the per-echo
  zero-crossing index i, not signal amplitude — there is no amplitude-ratio
  computation anywhere in features.py. This corrects an assumption in the
  original figure brief (and in CLAUDE.md's older Gruppe-7 description,
  which says "amplitude-based") against the actual implementation.
- G8 is verified to be a linear per-feature temperature detrend fit on
  NoFlow data: feature_tc = feature - alpha*(T_pipe - T_ref), applied to
  all 18 G5 components (norm_interval_up/down_0..8).

Run: python scripts/plot_thesis_feature_group_construction.py

Outputs:
    figures/feature_group_construction.{pdf,png}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =============================================================================
# STYLE (Okabe-Ito colour-blind-safe palette; matches plot_thesis_method_pipeline.py)
# =============================================================================

C_EDGE = '#202020'
C_TEXT = '#1a1a1a'
C_SUB = '#6e6e6e'
C_FILL = '#f4f4f5'

C_UP = '#0072B2'       # blue — upstream
C_DOWN = '#D55E00'     # vermillion — downstream
C_TEMP = '#CC79A7'     # reddish purple — temperature channel
C_HIGHLIGHT = '#009E73'  # bluish green — G5 "main encoder input"

LW_BOX = 1.2
LW_ARR = 1.15
LW_WAVE = 1.1

# Font/layout redesign for print legibility at ~0.9\linewidth (~5.49in on
# A4 with this thesis's 3.0/2.5cm margins). Two independent levers, tuned
# together rather than a flat font multiplier (which overflowed the boxes
# in an earlier pass -- uniformly scaling fonts AND geometry together is a
# no-op for print size, since LaTeX's downstream scale-to-\linewidth just
# undoes it; the only real lever is *disproportionate* box growth + gap
# reduction so boxes gain more room than fonts need, not the other way
# round). Font sizes below are chosen directly per tier (not from one
# global multiplier) so the smallest tier doesn't force every other tier
# to blow up past what its box can hold; each targets a specific print-pt
# size at the FIG_W chosen further down (see the printed scale-check at
# the bottom of this file).
TITLE_FS = 44.0
SUB_FS = 26.0
MATH_FS = 40.0
SECTION_FS = 26.0
NOTE_FS = 26.0

plt.rcParams.update({
    'font.size': 45.0,
    'text.color': C_TEXT,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})


# =============================================================================
# LAYOUT
# =============================================================================

Y_UP = 9.6
Y_DOWN = 1.4
AMP = 0.55

LEFT_X0 = 0.5
LEFT_W = 4.8

# Gaps carry no text, so they're cut hard -- every inch saved here goes
# straight to font-size headroom without touching any box's content room.
GAP1 = 0.5
MID_BOX_W = 3.5
MID_BOX_H = 1.9
MID_GAP = 0.35

mid_x0 = LEFT_X0 + LEFT_W + GAP1
mid_xs = [mid_x0 + i * (MID_BOX_W + MID_GAP) for i in range(3)]
mid_x_end = mid_xs[-1] + MID_BOX_W

GAP2 = 0.9
RIGHT_X0 = mid_x_end + GAP2
RIGHT_W = 9.3
RIGHT_H = 2.3
RIGHT_GAP = 1.15

FIG_W = RIGHT_X0 + RIGHT_W + 0.5
FIG_H = 13.0

LANE_TOP = Y_UP + 1.55
LANE_BOT = Y_DOWN - 1.55

# G-box vertical stack (top to bottom: G5, G6, G7, G8), centred between rows
stack_h = 4 * RIGHT_H + 3 * RIGHT_GAP
stack_top = (Y_UP + Y_DOWN) / 2 + stack_h / 2
g_ys = {
    'G5': stack_top - RIGHT_H,
    'G6': stack_top - RIGHT_H - (RIGHT_H + RIGHT_GAP),
    'G7': stack_top - RIGHT_H - 2 * (RIGHT_H + RIGHT_GAP),
    'G8': stack_top - RIGHT_H - 3 * (RIGHT_H + RIGHT_GAP),
}

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor('white')


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_box(x, y, w, h, title, subtitle=None, edgecolor=C_EDGE, fill=C_FILL,
             lw=LW_BOX, title_fs=TITLE_FS, sub_fs=SUB_FS, title_color=C_TEXT):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.10',
        linewidth=lw, edgecolor=edgecolor, facecolor=fill, zorder=3,
    )
    ax.add_patch(box)
    cx = x + w / 2
    if subtitle:
        ax.text(cx, y + h * 0.66, title, ha='center', va='center',
                 fontsize=title_fs, color=title_color, zorder=4)
        ax.text(cx, y + h * 0.27, subtitle, ha='center', va='center',
                 fontsize=sub_fs, color=C_SUB, zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha='center', va='center',
                 fontsize=title_fs, color=title_color, zorder=4)
    return box


def straight_arrow(p0, p1, color=C_EDGE, lw=LW_ARR, mscale=12, ls='-', zorder=2):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=mscale,
        linewidth=lw, color=color, linestyle=ls, zorder=zorder,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)


def curved_arrow(p0, p1, rad, color=C_EDGE, lw=LW_ARR, mscale=11, zorder=2, ls='-'):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=mscale,
        linewidth=lw, color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)


def draw_burst(x0, x1, y_base, color, n_cross=10, freq=5, decay=2.4, label=None):
    """Damped sinusoid with n_cross genuine zero crossings on [x0, x1];
    returns the x-positions of those crossings."""
    u = np.linspace(0, 1.05, 400)
    envelope = np.exp(-decay * u)
    wave = envelope * np.sin(2 * np.pi * freq * u)
    x = x0 + u * (x1 - x0)
    y = y_base + AMP * wave
    ax.plot(x, y, color=color, lw=LW_WAVE, alpha=0.85, zorder=3, solid_capstyle='round')

    cross_u = np.array([k / (2 * freq) for k in range(n_cross)])
    cross_x = x0 + cross_u * (x1 - x0)
    ax.scatter(cross_x, np.full(n_cross, y_base), s=20, color=color,
               edgecolor='white', linewidth=0.6, zorder=5)
    if label:
        ax.text(x0 - 0.15, y_base, label, ha='right', va='center',
                 fontsize=SECTION_FS - 1, color=color, fontweight='bold')
    return cross_x


def label_crossings(cross_x, y_level, sub, color, va):
    """Label crossing 0, ellipsis, 9 at an explicit y-level (only two indices
    labelled — adjacent full timestamp labels collide at this marker spacing,
    e.g. 't_down,0' / 't_down,1' — matches the prompt's own 't_x,0, ..., t_x,9'
    notation, which never wrote out index 1 either)."""
    ax.text(cross_x[0], y_level, rf'$t_{{{sub},0}}$', ha='center', va=va,
             fontsize=MATH_FS - 1.5, color=color)
    ax.text((cross_x[3] + cross_x[6]) / 2, y_level, r'$\cdots$', ha='center',
             va=va, fontsize=MATH_FS - 1.5, color=color)
    ax.text(cross_x[9], y_level, rf'$t_{{{sub},9}}$', ha='center', va=va,
             fontsize=MATH_FS - 1.5, color=color)


def interval_brackets(cross_x, y_line, label_y, sub, color, va):
    """One explicit interval bracket (0->1) + ellipsis, at explicit y-levels."""
    straight_arrow((cross_x[0], y_line), (cross_x[1], y_line),
                    color=color, lw=1.0, mscale=7, zorder=2)
    straight_arrow((cross_x[1], y_line), (cross_x[0], y_line),
                    color=color, lw=1.0, mscale=7, zorder=2)
    ax.text((cross_x[0] + cross_x[1]) / 2, label_y,
             rf'$\Delta_{{{sub},0}}$', ha='center', va=va,
             fontsize=MATH_FS - 2, color=color)
    ax.text((cross_x[2] + cross_x[4]) / 2, y_line, r'$\cdots\ \Delta_{' + sub + ',8}$',
             ha='left', va='center', fontsize=MATH_FS - 2, color=color)


# =============================================================================
# SECTION HEADERS
# =============================================================================

ax.text(LEFT_X0 + LEFT_W / 2, FIG_H - 0.35, 'Raw measurement cycle',
        ha='center', va='top', fontsize=SECTION_FS, fontweight='bold', color=C_TEXT)
ax.text((mid_x0 + mid_x_end) / 2, FIG_H - 0.35, 'Interval construction & normalization',
        ha='center', va='top', fontsize=SECTION_FS, fontweight='bold', color=C_TEXT)
ax.text(RIGHT_X0 + RIGHT_W / 2, FIG_H - 0.35, 'Feature groups',
        ha='center', va='top', fontsize=SECTION_FS, fontweight='bold', color=C_TEXT)

for xb in (LEFT_X0 + LEFT_W + GAP1 / 2, mid_x_end + GAP2 / 2):
    ax.plot([xb, xb], [0.5, FIG_H - 0.75], color='#d9d9d9', lw=0.9, zorder=0)

# =============================================================================
# LEFT: raw upstream / downstream bursts
# =============================================================================

up_cross = draw_burst(LEFT_X0, LEFT_X0 + LEFT_W, Y_UP, C_UP, label='Upstream')
down_cross = draw_burst(LEFT_X0, LEFT_X0 + LEFT_W, Y_DOWN, C_DOWN, label='Downstream')

# t-crossing labels face outward (away from the centre gap between the rows);
# interval brackets + notes face inward, stacked with explicit non-overlapping
# y-levels so the bracket line, its label, and the summary note never collide.
label_crossings(up_cross, Y_UP + AMP + 0.85, 'up', C_UP, va='bottom')
label_crossings(down_cross, Y_DOWN - AMP - 0.85, 'down', C_DOWN, va='top')

up_bracket_y = Y_UP - AMP - 0.85
up_label_y = up_bracket_y - 0.85
up_note_y = up_label_y - 0.85
interval_brackets(up_cross, up_bracket_y, up_label_y, 'up', C_UP, va='top')

down_bracket_y = Y_DOWN + AMP + 0.85
down_label_y = down_bracket_y + 0.85
down_note_y = down_label_y + 0.85
interval_brackets(down_cross, down_bracket_y, down_label_y, 'down', C_DOWN, va='bottom')

ax.text(LEFT_X0 + LEFT_W / 2, up_note_y,
        r'9 consecutive intervals $\Delta_{up,i}=t_{up,i+1}-t_{up,i}$, $i=0..8$',
        ha='center', va='top', fontsize=NOTE_FS, color=C_SUB, style='italic')
ax.text(LEFT_X0 + LEFT_W / 2, down_note_y,
        r'9 consecutive intervals $\Delta_{down,i}=t_{down,i+1}-t_{down,i}$, $i=0..8$',
        ha='center', va='bottom', fontsize=NOTE_FS, color=C_SUB, style='italic')

# =============================================================================
# MIDDLE: interval vector -> divide by span -> normalized vector (per direction)
# =============================================================================

up_box_y = Y_UP - MID_BOX_H / 2
down_box_y = Y_DOWN - MID_BOX_H / 2

draw_box(mid_xs[0], up_box_y, MID_BOX_W, MID_BOX_H,
         r'$\Delta_{up}\in\mathbb{R}^{9}$', edgecolor=C_UP)
draw_box(mid_xs[1], up_box_y, MID_BOX_W, MID_BOX_H,
         r'$\div\,S_{up}$', r'$S_{up}=t_{up,9}{-}t_{up,0}$', edgecolor=C_UP,
         title_fs=TITLE_FS + 1, sub_fs=NOTE_FS)
draw_box(mid_xs[2], up_box_y, MID_BOX_W, MID_BOX_H,
         r'$\hat\Delta_{up}\in\mathbb{R}^{9}$', edgecolor=C_UP)

draw_box(mid_xs[0], down_box_y, MID_BOX_W, MID_BOX_H,
         r'$\Delta_{down}\in\mathbb{R}^{9}$', edgecolor=C_DOWN)
draw_box(mid_xs[1], down_box_y, MID_BOX_W, MID_BOX_H,
         r'$\div\,S_{down}$', r'$S_{down}=t_{down,9}{-}t_{down,0}$', edgecolor=C_DOWN,
         title_fs=TITLE_FS + 1, sub_fs=NOTE_FS)
draw_box(mid_xs[2], down_box_y, MID_BOX_W, MID_BOX_H,
         r'$\hat\Delta_{down}\in\mathbb{R}^{9}$', edgecolor=C_DOWN)

for row_y, color in ((Y_UP, C_UP), (Y_DOWN, C_DOWN)):
    straight_arrow((LEFT_X0 + LEFT_W + 0.05, row_y), (mid_xs[0] - 0.05, row_y), color=color)
    for i in range(2):
        straight_arrow((mid_xs[i] + MID_BOX_W + 0.05, row_y), (mid_xs[i + 1] - 0.05, row_y), color=color)

# =============================================================================
# RIGHT: G5 / G6 / G7 / G8
# =============================================================================

g5_y, g6_y, g7_y, g8_y = g_ys['G5'], g_ys['G6'], g_ys['G7'], g_ys['G8']

draw_box(RIGHT_X0, g5_y, RIGHT_W, RIGHT_H,
         r'$G5 = [\hat\Delta_{up} \Vert \hat\Delta_{down}]$',
         'normalized directional intervals\n' + r'$\in\mathbb{R}^{18}$',
         edgecolor=C_HIGHLIGHT, lw=2.2, title_fs=TITLE_FS + 0.5)
ax.text(RIGHT_X0 + RIGHT_W / 2, g5_y - 0.6, 'Main encoder input',
        ha='center', va='top', fontsize=NOTE_FS + 0.3, color=C_HIGHLIGHT, fontweight='bold')

draw_box(RIGHT_X0, g6_y, RIGHT_W, RIGHT_H,
         r'$G6 = \hat\Delta_{up} - \hat\Delta_{down}$',
         'normalized directional differences\n' + r'$\in\mathbb{R}^{9}$')

draw_box(RIGHT_X0, g7_y, RIGHT_W, RIGHT_H,
         r'$G7_i = \Delta_{up,i}\,/\,\Delta_{down,i}$',
         'echo-ratio (per interval $i{=}0..8$)\n' + r'$\in\mathbb{R}^{9}$')
ax.text(RIGHT_X0 + RIGHT_W / 2, g7_y - 0.5,
        'interval ratio, not amplitude ratio (verified in code)',
        ha='center', va='top', fontsize=NOTE_FS - 4, color=C_SUB, style='italic')

draw_box(RIGHT_X0, g8_y, RIGHT_W, RIGHT_H,
         r'$G8 = \hat\Delta - \alpha\,(T_{pipe}{-}T_{ref})$',
         'temperature-corrected $G5$\n' + r'$\in\mathbb{R}^{18}$')

# G5 / G6: short elbows from the normalized-vector boxes (mid_xs[2])
norm_up_right = (mid_xs[2] + MID_BOX_W, Y_UP)
norm_down_right = (mid_xs[2] + MID_BOX_W, Y_DOWN)

for target_y, rad_up, rad_down in (
    (g5_y + RIGHT_H * 0.75, -0.28, -0.55),
    (g6_y + RIGHT_H * 0.5, -0.15, -0.30),
):
    curved_arrow(norm_up_right, (RIGHT_X0, target_y), rad=rad_up, color=C_UP)
    curved_arrow(norm_down_right, (RIGHT_X0, target_y), rad=rad_down, color=C_DOWN)

# G8: from normalized-vector boxes too, plus a temperature input
curved_arrow(norm_up_right, (RIGHT_X0, g8_y + RIGHT_H * 0.72), rad=0.55, color=C_UP)
curved_arrow(norm_down_right, (RIGHT_X0, g8_y + RIGHT_H * 0.5), rad=0.12, color=C_DOWN)

temp_box_w, temp_box_h = 4.6, 2.3
temp_x = mid_xs[2] - 0.1
temp_y = g8_y - 2.95
draw_box(temp_x, temp_y, temp_box_w, temp_box_h,
         r'$T_{pipe},\,T_{amb}$', 'measured alongside\ncycle',
         edgecolor=C_TEMP, fill='white', title_fs=MATH_FS - 2, sub_fs=NOTE_FS - 2)
straight_arrow((temp_x + temp_box_w, temp_y + temp_box_h / 2),
               (RIGHT_X0, g8_y + RIGHT_H * 0.25), color=C_TEMP)

# G7: long-distance elbow routed around the outside (above upstream row,
# below downstream row), staying in the gutter strictly LEFT of the G-box
# column (x < RIGHT_X0) the whole way, then a short final hop into G7's
# left edge — this visually skips the div / normalized boxes entirely
# instead of passing behind them, reflecting that echo_ratio uses RAW
# intervals, not normalized ones.
interval_up_right = (mid_xs[0] + MID_BOX_W, Y_UP)
interval_down_right = (mid_xs[0] + MID_BOX_W, Y_DOWN)
LANE_X = RIGHT_X0 - 0.45
g7_cy = g7_y + RIGHT_H / 2

ax.plot([interval_up_right[0], interval_up_right[0] + 0.3], [Y_UP, LANE_TOP],
        color=C_UP, lw=LW_ARR, zorder=2)
ax.plot([interval_up_right[0] + 0.3, LANE_X], [LANE_TOP, LANE_TOP],
        color=C_UP, lw=LW_ARR, zorder=2)
ax.plot([LANE_X, LANE_X], [LANE_TOP, g7_cy + 0.18],
        color=C_UP, lw=LW_ARR, zorder=2)
straight_arrow((LANE_X, g7_cy + 0.18), (RIGHT_X0, g7_cy + 0.18), color=C_UP)

ax.plot([interval_down_right[0], interval_down_right[0] + 0.3], [Y_DOWN, LANE_BOT],
        color=C_DOWN, lw=LW_ARR, zorder=2)
ax.plot([interval_down_right[0] + 0.3, LANE_X - 0.15], [LANE_BOT, LANE_BOT],
        color=C_DOWN, lw=LW_ARR, zorder=2)
ax.plot([LANE_X - 0.15, LANE_X - 0.15], [LANE_BOT, g7_cy - 0.18],
        color=C_DOWN, lw=LW_ARR, zorder=2)
straight_arrow((LANE_X - 0.15, g7_cy - 0.18), (RIGHT_X0, g7_cy - 0.18), color=C_DOWN)

# =============================================================================
# SAVE
# =============================================================================

out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
for ext, dpi in [('pdf', None), ('png', 300)]:
    p = out_dir / f'feature_group_construction.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved: {p}')
plt.close(fig)

# =============================================================================
# TERMINAL REPRODUCIBILITY REPORT
# =============================================================================

print('\n=== Feature group construction — reproducibility report ===')
print('  Exact input columns used:')
print('    raw upstream ZC times:   ' + ', '.join(f't_up_{i} in ps' for i in range(10)))
print('    raw downstream ZC times: ' + ', '.join(f't_down_{i} in ps' for i in range(10)))
print('    temperature (G8 only):   Temp Pipe in °C')
print()
print('  Verified G7 equation (src/preparation/features.py, compute_features):')
print('    echo_ratio_i = interval_up_i / interval_down_i')
print('                 = (t_up_{i+1} - t_up_i) / (t_down_{i+1} - t_down_i),  i = 0..8')
print('    NOTE: interval-based, NOT amplitude-based. No amplitude-ratio computation')
print('    exists in features.py; this corrects the "amplitude-based" description')
print('    in the original figure brief / CLAUDE.md Gruppe-7 text.')
print()
print('  Verified G8 correction model (src/preparation/features.py,')
print('  fit_temperature_correction + apply_temperature_correction):')
print('    fit on NoFlow segments:  feature = alpha * T_pipe + beta')
print('      alpha = cov(feature, T_pipe) / var(T_pipe),  T_ref = mean(T_pipe | NoFlow)')
print('    applied to all segments: feature_tc = feature - alpha * (T_pipe - T_ref)')
print('    Applied to all 18 G5 components (norm_interval_up/down_0..8) to form G8.')

tc_coeff_path = Path(__file__).resolve().parent.parent / 'data' / 'prepared' / 'features' / 'temperature_correction.json'
if tc_coeff_path.exists():
    with open(tc_coeff_path) as f:
        coeffs = json.load(f)
    print(f'    Fitted coefficients found: {tc_coeff_path}')
    for k in list(coeffs)[:3]:
        c = coeffs[k]
        print(f'      {k}: alpha={c["alpha"]:.4g}, T_ref={c["T_ref"]:.3g}°C')
    if len(coeffs) > 3:
        print(f'      ... ({len(coeffs)} features total)')
else:
    print(f'    No fitted-coefficient file found at {tc_coeff_path} — model shown structurally only.')

print()
print('  Dimensions:')
print('    G5 (normalized directional intervals):   18')
print('    G6 (normalized directional differences):  9')
print('    G7 (echo-ratio):                          9')
print('    G8 (temperature-corrected G5):            18')
print()
print('  Conceptual-only branches: NONE — all four formulas were verified directly')
print('  from src/preparation/features.py; nothing in this figure was invented.')
