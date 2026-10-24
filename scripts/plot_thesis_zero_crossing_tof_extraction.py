#!/usr/bin/env python
"""
Thesis figure — ToF zero-crossing feature extraction (detailed, technical version).

Outputs:
  figures/zero_crossing_tof_extraction.pdf
  figures/zero_crossing_tof_extraction.png

Run: python scripts/plot_thesis_zero_crossing_tof_extraction.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── colours ─────────────────────────────────────────────────────────────────
C_DARK  = '#1a1a1a'
C_BLUE  = '#1f6fb2'   # received waveform
C_RED   = '#c0392b'   # first-hit threshold
C_GRN   = '#276419'   # selected zero crossings + TDC events
C_GRAY  = '#555555'
C_LGRAY = '#cccccc'

# ─── line weights ────────────────────────────────────────────────────────────
LW  = 1.7   # waveform / fire burst
LWT = 1.0   # thin structural lines
LWD = 0.75  # dashed lines

# ─── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7.2))
ax.set_xlim(-0.1, 16.8)
ax.set_ylim(-3.2, 11.2)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ─── lane centres ────────────────────────────────────────────────────────────
Y_FIRE = 9.2
Y_WAVE = 5.3
Y_TDC  = 1.0

# ─── waveform parameters ─────────────────────────────────────────────────────
X_SEP         = 1.15
X_BURST_START = 1.20
X_BURST_END   = 2.55
X_WAVE_START  = 3.20
X_WAVE_END    = 15.50

FREQ       = 0.80
AMP        = 1.20    # rising max amplitude
AMP_STABLE = 1.72    # stable amplitude (larger)
T_RISE     = 2.50    # tau where rising ends   (4/(2×0.80)=2.5 → zero crossing ✓)
T_STABLE_E = 7.50    # tau where stable ends   (12/(2×0.80)=7.5 → zero crossing ✓)
T_DUR      = X_WAVE_END - X_WAVE_START   # 12.3

tau  = np.linspace(0, T_DUR, 15000)
x_wv = tau + X_WAVE_START

# ── Amplitude envelope ────────────────────────────────────────────────────────
# Rising: same as existing figure
rising_env = AMP * (1.0 - np.exp(-tau * 1.65))

# Stable: AMP_STABLE with very mild deterministic variation (peaks not all equal)
stable_var = (0.045 * np.sin(0.38 * tau + 1.20)
            - 0.030 * np.cos(0.72 * tau + 2.10))
stable_env = AMP_STABLE * (1.0 + stable_var)

# Decay: exponential from AMP_STABLE
decay_env = AMP_STABLE * np.exp(-(tau - T_STABLE_E) * 0.50)

# Hard-cut at T_RISE and T_STABLE_E (both are zero crossings → no visible kink)
env = np.where(
    tau < T_RISE,
    rising_env,
    np.where(tau < T_STABLE_E, stable_env, decay_env)
)

y_wv = Y_WAVE + env * np.sin(2 * np.pi * FREQ * tau)

# ── First-hit threshold ───────────────────────────────────────────────────────
FH_AMP = 0.68
y_fh   = Y_WAVE + FH_AMP

sig   = env * np.sin(2 * np.pi * FREQ * tau)
above = sig > FH_AMP
in_ri = tau < T_RISE + 1.0
idx   = np.where(above & in_ri)[0]
fh_tau = tau[idx[0]]        if len(idx) else T_RISE
fh_x   = fh_tau + X_WAVE_START
fh_y   = Y_WAVE + sig[idx[0]] if len(idx) else y_fh

# ── Selected zero crossings (4, in mid-stable) ────────────────────────────────
# tau_zc = n/(2*FREQ) for n = 5,6,7,8 → tau = 3.125, 3.75, 4.375, 5.0
N_SEL  = 4
n_base = 5
sel_tau = np.array([(n_base + k) / (2 * FREQ) for k in range(N_SEL)])
sel_x   = sel_tau + X_WAVE_START

# ─────────────────────────────────────────────────────────────────────────────
# Left separator + row labels
# ─────────────────────────────────────────────────────────────────────────────
ax.plot([X_SEP, X_SEP], [-2.8, 11.0], color=C_LGRAY, lw=0.7)

lkw = dict(ha='right', va='center', fontsize=10.5, color=C_DARK)
ax.text(X_SEP - 0.10, Y_FIRE,        'Fire burst',           **lkw)
ax.text(X_SEP - 0.10, Y_WAVE,        'Received\nwave',       **lkw)
ax.text(X_SEP - 0.10, Y_TDC + 0.20, 'TDC stop\nevents',     **lkw)

# Subtle lane dividers
for y_div in [Y_FIRE - 0.85, Y_TDC + 0.65]:
    ax.plot([X_SEP + 0.04, X_WAVE_END + 0.3], [y_div, y_div],
            color=C_LGRAY, lw=0.45, alpha=0.65)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Fire burst
# ═════════════════════════════════════════════════════════════════════════════
FIRE_H = 0.46
N_PUL  = 4
pw     = (X_BURST_END - X_BURST_START) / (2 * N_PUL)

bx, by = [X_BURST_START], [Y_FIRE - FIRE_H]
for i in range(N_PUL):
    x0 = X_BURST_START + 2 * i * pw
    bx += [x0, x0, x0+pw, x0+pw, x0+2*pw]
    by += [Y_FIRE-FIRE_H, Y_FIRE+FIRE_H, Y_FIRE+FIRE_H, Y_FIRE-FIRE_H, Y_FIRE-FIRE_H]
bx.append(X_BURST_END); by.append(Y_FIRE - FIRE_H)
ax.plot(bx, by, color=C_DARK, lw=LW, solid_capstyle='butt', solid_joinstyle='miter')

# Baseline stubs around burst
for xa, xb in [(X_SEP+0.06, X_BURST_START), (X_BURST_END, X_BURST_END+0.3)]:
    ax.plot([xa, xb], [Y_FIRE-FIRE_H]*2, color=C_DARK, lw=LWT*0.6, alpha=0.4)

# ═════════════════════════════════════════════════════════════════════════════
# 2. Received waveform (blue)
# ═════════════════════════════════════════════════════════════════════════════
ax.plot(x_wv, y_wv, color=C_BLUE, lw=LW, zorder=3)

# Region brackets above waveform
x_rise_end   = X_WAVE_START + T_RISE    # 5.70
x_stable_end = X_WAVE_START + T_STABLE_E  # 10.70
Y_BR  = Y_WAVE + AMP_STABLE + 0.55
Y_LBL = Y_BR + 0.32

for xv in [X_WAVE_START, x_rise_end, x_stable_end, X_WAVE_END]:
    ax.plot([xv, xv], [Y_WAVE + AMP_STABLE + 0.18, Y_BR], color=C_LGRAY, lw=0.8)
for xa, xb in [(X_WAVE_START, x_rise_end),
               (x_rise_end,   x_stable_end),
               (x_stable_end, X_WAVE_END)]:
    ax.plot([xa, xb], [Y_BR, Y_BR], color=C_LGRAY, lw=0.8)
for (xa, xb), lbl in [
    ((X_WAVE_START, x_rise_end),   'Rising'),
    ((x_rise_end,   x_stable_end), 'Stable'),
    ((x_stable_end, X_WAVE_END),   'Decaying'),
]:
    ax.text((xa+xb)/2, Y_LBL, lbl, ha='center', va='bottom',
            fontsize=9.5, color=C_GRAY, style='italic')

# ═════════════════════════════════════════════════════════════════════════════
# 3. Comparator reference: first-hit threshold → step → zero ref
# ═════════════════════════════════════════════════════════════════════════════

# Faint zero-reference before first-hit (dotted, before detection)
ax.plot([X_WAVE_START, fh_x], [Y_WAVE, Y_WAVE],
        color=C_GRAY, lw=LWD * 0.6, ls=':', alpha=0.35, zorder=2)

# First-hit threshold segment (rising region only)
ax.plot([X_WAVE_START - 0.05, fh_x], [y_fh, y_fh],
        color=C_RED, lw=LWT + 0.5, zorder=4)

# Vertical step from y_fh → Y_WAVE at first-hit x
ax.plot([fh_x, fh_x], [y_fh, Y_WAVE],
        color=C_RED, lw=LWT, ls='-', alpha=0.75, zorder=4)

# Zero-crossing reference after step (dashed gray)
ax.plot([fh_x, X_WAVE_END + 0.22], [Y_WAVE, Y_WAVE],
        color=C_GRAY, lw=LWD, ls='--', alpha=0.60, zorder=2)

# First-hit marker: filled downward triangle
ax.plot(fh_x, fh_y, 'v', color=C_RED, ms=9, zorder=8,
        markerfacecolor=C_RED, markeredgecolor='white', markeredgewidth=0.6)

# Labels
ax.text(X_WAVE_START, y_fh + 0.26,
        'First-hit level', ha='left', va='bottom', fontsize=9.5, color=C_RED)
ax.text(X_WAVE_END + 0.30, Y_WAVE,
        'zero ref.', ha='left', va='center', fontsize=8.5, color=C_GRAY, style='italic')

# ═════════════════════════════════════════════════════════════════════════════
# 4. Selected zero crossings, guide lines, ZC labels
# ═════════════════════════════════════════════════════════════════════════════
TDC_H = 0.42
TDC_W = 0.10

for k, xi in enumerate(sel_x):
    # Guide line (light dashed, full height from waveform to TDC)
    ax.plot([xi, xi], [Y_WAVE - 0.22, Y_TDC + TDC_H + 0.10],
            color=C_LGRAY, lw=1.0, ls='--', zorder=2)

    # Hollow circle marker on waveform at zero-crossing
    ax.plot(xi, Y_WAVE, 'o', color=C_GRN, ms=7.5, zorder=7,
            markerfacecolor='white', markeredgewidth=2.2)

    # ZC label just above waveform zero-reference
    ax.text(xi, Y_WAVE + 0.28,
            f'$z_{{{k+1}}}$',
            ha='center', va='bottom', fontsize=9, color=C_GRN)

    # TDC pulse (narrow filled rectangle)
    rect = mpatches.Rectangle(
        (xi - TDC_W, Y_TDC), 2*TDC_W, TDC_H,
        facecolor=C_GRN, edgecolor='white', linewidth=0.6,
        alpha=0.88, zorder=5)
    ax.add_patch(rect)

    # TDC label below pulse
    ax.text(xi, Y_TDC - 0.28,
            f'$z_{{{k+1}}}$',
            ha='center', va='top', fontsize=8, color=C_GRN)

# TDC baseline
ax.plot([X_SEP+0.05, X_WAVE_END+0.22], [Y_TDC, Y_TDC],
        color=C_DARK, lw=0.65, alpha=0.40)

# "ToF start" annotation on first TDC pulse
ax.annotate('ToF\nstart',
            xy=(sel_x[0], Y_TDC + TDC_H),
            xytext=(sel_x[0] - 0.80, Y_TDC + TDC_H + 0.70),
            fontsize=8.5, color=C_GRN, ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color=C_GRN,
                            lw=0.9, mutation_scale=8))

# "Selected zero crossings" label to the right of last marker
ax.text(sel_x[-1] + 0.30, Y_WAVE + 0.30,
        'Selected\nzero crossings',
        ha='left', va='bottom', fontsize=9, color=C_GRN)

# ═════════════════════════════════════════════════════════════════════════════
# 5. DELREL bracket (first-hit → first selected ZC)
# ═════════════════════════════════════════════════════════════════════════════
Y_DR     = Y_WAVE - 1.50   # bracket y (below waveform trough)
Y_DR_TK  = 0.18            # tick height above/below bracket line

# Horizontal double-headed arrow
ax.annotate('', xy=(sel_x[0], Y_DR), xytext=(fh_x, Y_DR),
            arrowprops=dict(arrowstyle='<->', color=C_RED,
                            lw=1.1, mutation_scale=8))

# Vertical tick connectors (dashed)
for xi in (fh_x, sel_x[0]):
    ax.plot([xi, xi], [Y_WAVE - 0.1, Y_DR + Y_DR_TK],
            color=C_RED, lw=0.75, ls=':', alpha=0.55)

ax.text((fh_x + sel_x[0]) / 2, Y_DR - 0.28,
        'DELREL offset',
        ha='center', va='top', fontsize=9, color=C_RED)

# ═════════════════════════════════════════════════════════════════════════════
# 6. Time axis arrow
# ═════════════════════════════════════════════════════════════════════════════
ax.annotate('', xy=(16.2, -2.7), xytext=(13.5, -2.7),
            arrowprops=dict(arrowstyle='->', color=C_DARK,
                            lw=1.3, mutation_scale=11))
ax.text(16.35, -2.7, '$t$', ha='left', va='center',
        fontsize=12, color=C_DARK)

# ─── save ─────────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 300)]:
    p = out_dir / f'zero_crossing_tof_extraction.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
