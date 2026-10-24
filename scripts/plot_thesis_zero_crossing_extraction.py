#!/usr/bin/env python
"""
Thesis figure — Zero-crossing-based time-of-flight feature extraction schematic.

Outputs:
  figures/zero_crossing_feature_extraction.pdf
  figures/zero_crossing_feature_extraction.png

Run: python scripts/plot_thesis_zero_crossing_extraction.py
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
C_GRAY  = '#777777'
C_LGRAY = '#cccccc'
C_RED   = '#c0392b'   # first-hit threshold
C_BLUE  = '#2166ac'   # zero crossings + TDC events

# ─── global line weights ─────────────────────────────────────────────────────
LW_MAIN = 1.6
LW_THIN = 0.9
LW_DASH = 0.8

# ─── figure canvas ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5.8))
ax.set_xlim(0.0, 14.2)
ax.set_ylim(-1.5, 9.2)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ─── lane y-centres ──────────────────────────────────────────────────────────
Y_FIRE = 7.5    # fire burst
Y_WAVE = 3.9    # received waveform
Y_TDC  = 0.75   # TDC timing events

# ─── waveform geometry ───────────────────────────────────────────────────────
X_BURST_START = 1.15
X_BURST_END   = 2.35
X_WAVE_START  = 3.05   # received wave begins after propagation gap
X_WAVE_END    = 13.05

FREQ = 0.80    # cycles per length unit
AMP  = 1.20    # max amplitude (y-units)
T_RISE       = 2.5    # rising phase duration (in tau)
T_STABLE_END = 7.2    # stable phase end (in tau)
T_DUR        = X_WAVE_END - X_WAVE_START   # 10.0

tau  = np.linspace(0, T_DUR, 12000)
x_wv = tau + X_WAVE_START

# Amplitude envelope
# Rising: unchanged (tau=T_RISE=2.5 is exactly a zero crossing of sin at FREQ=0.80,
# so the envelope step there is invisible in the waveform)
AMP_STABLE = 1.72   # visibly larger stable amplitude
env = np.where(
    tau < T_RISE,
    AMP * (1.0 - np.exp(-tau * 1.65)),          # rising: exactly as before
    np.where(
        tau < T_STABLE_END,
        AMP_STABLE,                               # stable: larger peaks
        AMP_STABLE * np.exp(-(tau - T_STABLE_END) * 0.52)  # decay from larger amp
    )
)
y_wv = Y_WAVE + env * np.sin(2 * np.pi * FREQ * tau)

# ─── first-hit threshold ─────────────────────────────────────────────────────
FH_AMP = 0.68
y_fh   = Y_WAVE + FH_AMP

sig_vals = env * np.sin(2 * np.pi * FREQ * tau)
rising   = tau < T_RISE + 1.0
above    = sig_vals > FH_AMP
first_ab = np.where(above & rising)[0]
fh_tau   = tau[first_ab[0]]        if len(first_ab) else T_RISE
fh_x     = fh_tau + X_WAVE_START
fh_y     = Y_WAVE + sig_vals[first_ab[0]] if len(first_ab) else y_fh

# ─── selected zero crossings (stable region) ─────────────────────────────────
n_lo = int(np.ceil((T_RISE + 0.3) * 2 * FREQ))
n_hi = int(np.floor((T_STABLE_END - 0.3) * 2 * FREQ))
zc_tau = np.array([n / (2 * FREQ) for n in range(n_lo, n_hi + 1)])
zc_x   = zc_tau + X_WAVE_START
# keep at most 8
zc_x   = zc_x[:8]

# ═════════════════════════════════════════════════════════════════════════════
# Row separator + left-margin labels
# ═════════════════════════════════════════════════════════════════════════════
X_SEP = 1.00
ax.plot([X_SEP, X_SEP], [-1.0, 9.0], color=C_LGRAY, lw=0.7)

lkw = dict(ha='right', va='center', fontsize=9.5, color=C_DARK)
ax.text(X_SEP - 0.08, Y_FIRE,        'Fire burst',         **lkw)
ax.text(X_SEP - 0.08, Y_WAVE,        'Received\nwave',     **lkw)
ax.text(X_SEP - 0.08, Y_TDC + 0.17, 'TDC timing\nevents', **lkw)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Fire burst — square pulses
# ═════════════════════════════════════════════════════════════════════════════
FIRE_H  = 0.44
N_PULSE = 4
pw = (X_BURST_END - X_BURST_START) / (2 * N_PULSE)

bx = [X_BURST_START]
by = [Y_FIRE - FIRE_H]
for i in range(N_PULSE):
    x0 = X_BURST_START + 2 * i * pw
    bx += [x0,            x0,            x0 + pw,      x0 + pw,      x0 + 2 * pw]
    by += [Y_FIRE - FIRE_H, Y_FIRE + FIRE_H, Y_FIRE + FIRE_H, Y_FIRE - FIRE_H, Y_FIRE - FIRE_H]
bx.append(X_BURST_END)
by.append(Y_FIRE - FIRE_H)

ax.plot(bx, by, color=C_DARK, lw=LW_MAIN, solid_capstyle='butt',
        solid_joinstyle='miter')

# Short baselines on either side of burst
ax.plot([X_SEP + 0.04, X_BURST_START], [Y_FIRE - FIRE_H] * 2,
        color=C_DARK, lw=LW_THIN * 0.7, alpha=0.5)
ax.plot([X_BURST_END, X_BURST_END + 0.25], [Y_FIRE - FIRE_H] * 2,
        color=C_DARK, lw=LW_THIN * 0.7, alpha=0.5)

# ═════════════════════════════════════════════════════════════════════════════
# 2. Received waveform
# ═════════════════════════════════════════════════════════════════════════════
ax.plot(x_wv, y_wv, color=C_DARK, lw=LW_MAIN, zorder=3)

# Zero-crossing reference (dashed)
ax.plot([X_WAVE_START - 0.1, X_WAVE_END + 0.15],
        [Y_WAVE, Y_WAVE],
        color=C_GRAY, lw=LW_DASH, ls='--', alpha=0.65, zorder=2)
ax.text(X_WAVE_END + 0.22, Y_WAVE, 'zero ref.',
        ha='left', va='center', fontsize=8, color=C_GRAY, style='italic')

# ── Region brackets above waveform ───────────────────────────────────────────
x_rise_end   = X_WAVE_START + T_RISE          # 5.55
x_stable_end = X_WAVE_START + T_STABLE_END    # 10.25

Y_BR  = Y_WAVE + AMP + 0.42
Y_LBL = Y_BR + 0.30

for xv in [X_WAVE_START, x_rise_end, x_stable_end, X_WAVE_END]:
    ax.plot([xv, xv], [Y_WAVE + AMP + 0.12, Y_BR], color=C_LGRAY, lw=0.75)
for (xa, xb) in [(X_WAVE_START, x_rise_end),
                  (x_rise_end, x_stable_end),
                  (x_stable_end, X_WAVE_END)]:
    ax.plot([xa, xb], [Y_BR, Y_BR], color=C_LGRAY, lw=0.75)

for (xa, xb), lbl in [
    ((X_WAVE_START, x_rise_end),   'Rising'),
    ((x_rise_end,   x_stable_end), 'Stable'),
    ((x_stable_end, X_WAVE_END),   'Decaying'),
]:
    ax.text((xa + xb) / 2, Y_LBL, lbl,
            ha='center', va='bottom', fontsize=9,
            color=C_GRAY, style='italic')

# ═════════════════════════════════════════════════════════════════════════════
# 3. First-hit threshold
# ═════════════════════════════════════════════════════════════════════════════
ax.plot([X_WAVE_START - 0.05, fh_x], [y_fh, y_fh],
        color=C_RED, lw=LW_THIN + 0.4, zorder=4)
ax.plot(fh_x, fh_y, 'v', color=C_RED, ms=7.5, zorder=7)
ax.text(X_WAVE_START, y_fh + 0.22, 'First-hit level',
        ha='left', va='bottom', fontsize=9, color=C_RED)

# ═════════════════════════════════════════════════════════════════════════════
# 4. Selected zero crossings + vertical drops + TDC pulses
# ═════════════════════════════════════════════════════════════════════════════
TDC_H = 0.36
TDC_W = 0.08

for xi in zc_x:
    # Hollow circle marker on waveform at Y_WAVE
    ax.plot(xi, Y_WAVE, 'o', color=C_BLUE, ms=5.5, zorder=6,
            markerfacecolor='white', markeredgewidth=1.7)
    # Dotted vertical drop
    ax.plot([xi, xi], [Y_WAVE - 0.18, Y_TDC + TDC_H + 0.06],
            color=C_BLUE, lw=0.85, ls=':', alpha=0.65, zorder=2)
    # TDC rectangular pulse
    rect = mpatches.Rectangle(
        (xi - TDC_W, Y_TDC), 2 * TDC_W, TDC_H,
        facecolor=C_BLUE, edgecolor=C_BLUE, linewidth=0.5,
        alpha=0.88, zorder=4
    )
    ax.add_patch(rect)

# TDC baseline
ax.plot([X_SEP + 0.04, X_WAVE_END + 0.15], [Y_TDC, Y_TDC],
        color=C_DARK, lw=0.75, alpha=0.45)

# ── Annotations ──────────────────────────────────────────────────────────────
# "Selected zero crossings" — point to 3rd crossing from outside waveform area
if len(zc_x) >= 3:
    ann_xi = zc_x[2]
    ax.annotate(
        'Selected\nzero crossings',
        xy=(ann_xi, Y_WAVE),
        xytext=(ann_xi - 1.6, Y_WAVE - 1.55),
        fontsize=8.5, color=C_BLUE, ha='center', va='top',
        arrowprops=dict(arrowstyle='->', color=C_BLUE,
                        lw=0.95, mutation_scale=9,
                        connectionstyle='arc3,rad=0.30')
    )

# "TDC timing events" — to the right of last pulse
ax.text(X_WAVE_END + 0.22, Y_TDC + TDC_H / 2,
        'TDC timing events',
        ha='left', va='center', fontsize=9, color=C_BLUE)

# ═════════════════════════════════════════════════════════════════════════════
# Time axis arrow (bottom right)
# ═════════════════════════════════════════════════════════════════════════════
ax.annotate('', xy=(13.95, -1.0), xytext=(11.5, -1.0),
            arrowprops=dict(arrowstyle='->', color=C_DARK,
                            lw=1.2, mutation_scale=10))
ax.text(14.05, -1.0, '$t$', ha='left', va='center',
        fontsize=11, color=C_DARK)

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
if os.getenv('DOCKER_ENV'):
    out_dir = Path('/app/images/thesis')
else:
    out_dir = Path(__file__).resolve().parent.parent / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

for ext, dpi in [('pdf', None), ('png', 300)]:
    p = out_dir / f'zero_crossing_feature_extraction.{ext}'
    kw = dict(format=ext, bbox_inches='tight')
    if dpi:
        kw['dpi'] = dpi
    fig.savefig(p, **kw)
    print(f'Saved → {p}')

plt.close(fig)
