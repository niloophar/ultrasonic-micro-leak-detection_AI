#!/usr/bin/env python
"""
Thesis figure — Normalized interval features from zero-crossing timestamps.

Output: figures/normalised_interval_features.pdf

Run locally or on server:
    python scripts/plot_thesis_normalised_interval_features.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── timestamps (slightly uneven) ──────────────────────────────────────────────
t = np.array([0.0, 1.1, 2.3, 3.2, 4.6, 5.5, 6.8, 7.9, 8.7, 10.0])
intervals = np.diff(t)
total_span = t[-1] - t[0]
ni = intervals / total_span

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.18)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── colours ───────────────────────────────────────────────────────────────────
C_AXIS    = '#222222'
C_TICK    = '#222222'
C_BRACK   = '#555555'
C_TOTAL   = '#1a6599'
C_BAR     = '#4a90c4'
C_UNIFORM = '#aaaaaa'

# ════════════════════════════════════════════════════════════════════════════
# Panel A — Selected zero-crossing timestamps
# ════════════════════════════════════════════════════════════════════════════
ax1.set_title('(A)  Selected zero-crossing timestamps',
              fontsize=12, fontweight='bold', loc='left', pad=8)
ax1.axis('off')
ax1.set_xlim(-0.7, 11.2)
ax1.set_ylim(-1.75, 2.0)

Y_AXIS     =  0.0
Y_TICK     =  0.55   # top of short tick marks
Y_BR_HI    =  1.20   # bracket row for even-indexed intervals
Y_BR_LO    =  0.85   # bracket row for odd-indexed intervals
Y_LBL_T    = -0.28   # t_i labels
Y_TOTAL    = -1.00   # total-span bracket
Y_TOTAL_L  = -1.42   # total-span label

# Time axis
ax1.annotate('', xy=(11.0, Y_AXIS), xytext=(-0.5, Y_AXIS),
             arrowprops=dict(arrowstyle='->', color=C_AXIS, lw=1.8,
                             mutation_scale=12))
ax1.text(11.15, Y_AXIS, 'time', ha='left', va='center',
         fontsize=9, color=C_AXIS, style='italic')

# Tick marks and labels
for i, ti in enumerate(t):
    ax1.plot([ti, ti], [Y_AXIS, Y_TICK], color=C_TICK, lw=1.6,
             solid_capstyle='round')
    ax1.plot(ti, Y_AXIS, 'o', color=C_TICK, ms=4.5, zorder=5)
    ax1.text(ti, Y_LBL_T, f'$t_{{{i}}}$',
             ha='center', va='top', fontsize=9.5)

# Interval brackets — alternating heights, label a few
LABEL_IDXS = {0, 1, 7, 8}
for i in range(9):
    x1, x2 = t[i], t[i + 1]
    xm = (x1 + x2) / 2
    yb = Y_BR_HI if i % 2 == 0 else Y_BR_LO

    # Double-headed arrow
    ax1.annotate('', xy=(x2, yb), xytext=(x1, yb),
                 arrowprops=dict(arrowstyle='<->', color=C_BRACK,
                                 lw=1.1, mutation_scale=7))

    # Dotted connectors from tick tops to bracket
    for xi in (x1, x2):
        ax1.plot([xi, xi], [Y_TICK, yb], color=C_BRACK,
                 lw=0.7, ls=':', alpha=0.55)

    # Label
    y_text = yb + 0.14
    if i in LABEL_IDXS:
        ax1.text(xm, y_text, f'$\\Delta t_{{{i}}}$',
                 ha='center', va='bottom', fontsize=8.5, color=C_BRACK)
    elif i == 4:
        ax1.text(xm, y_text, r'$\cdots$',
                 ha='center', va='bottom', fontsize=10, color=C_BRACK)

# Total span bracket
ax1.annotate('', xy=(t[-1], Y_TOTAL), xytext=(t[0], Y_TOTAL),
             arrowprops=dict(arrowstyle='<->', color=C_TOTAL,
                             lw=1.6, mutation_scale=9))
for xi in (t[0], t[-1]):
    ax1.plot([xi, xi], [Y_AXIS - 0.06, Y_TOTAL],
             color=C_TOTAL, lw=0.9, ls='--', alpha=0.65)
ax1.text((t[0] + t[-1]) / 2, Y_TOTAL_L,
         r'total span $= t_9 - t_0$',
         ha='center', va='top', fontsize=9.5, color=C_TOTAL,
         fontweight='bold')

# ════════════════════════════════════════════════════════════════════════════
# Panel B — Normalized interval profile
# ════════════════════════════════════════════════════════════════════════════
ax2.set_title('(B)  Normalized interval profile',
              fontsize=12, fontweight='bold', loc='left', pad=8)

x_pos = np.arange(9)
ax2.bar(x_pos, ni, color=C_BAR, edgecolor='white', linewidth=0.7,
        alpha=0.88, width=0.65, zorder=3)

# Uniform-spacing reference line
ax2.axhline(1 / 9, color=C_UNIFORM, lw=1.2, ls='--', zorder=2)
ax2.text(8.45, 1 / 9 + 0.004, 'uniform', ha='right', va='bottom',
         fontsize=8, color='#888888')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(
    [r'$\mathrm{NI}_{' + str(i) + r'}$' for i in range(9)],
    fontsize=9.5)
ax2.set_ylabel('Relative interval length', fontsize=10)
ax2.set_ylim(0, max(ni) * 1.55)
ax2.spines[['top', 'right']].set_visible(False)
ax2.tick_params(axis='y', labelsize=9)
ax2.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

# Formula box
ax2.text(0.985, 0.95,
         r'$\mathrm{NI}_i = \dfrac{t_{i+1} - t_i}{t_9 - t_0}$',
         transform=ax2.transAxes, ha='right', va='top', fontsize=12,
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#f4f4f4',
                   edgecolor='#cccccc', alpha=0.95))

# ── save ──────────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_path = Path('/app/images/thesis/normalised_interval_features.pdf')
else:
    out_path = (Path(__file__).resolve().parent.parent
                / 'figures' / 'normalised_interval_features.pdf')

out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
print(f'Saved → {out_path}')
plt.close(fig)
