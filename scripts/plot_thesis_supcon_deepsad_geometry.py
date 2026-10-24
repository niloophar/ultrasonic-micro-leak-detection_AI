#!/usr/bin/env python
"""
Thesis figure — SupCon vs DeepSAD latent-space geometry (conceptual).

Output: figures/supcon_deepsad_geometry.pdf

Run locally or on server:
    python scripts/plot_thesis_supcon_deepsad_geometry.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── reproducible ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── colours ───────────────────────────────────────────────────────────────────
C_NF  = '#2166ac'   # NoFlow — blue
C_LK  = '#d6604d'   # Leakage — red-orange
C_CTR = '#1a9850'   # DeepSAD centre — green
C_MRG = '#888888'   # margin circle — grey

# ── data ──────────────────────────────────────────────────────────────────────
# SupCon: two well-separated clusters
nf_sc = rng.normal(loc=[-2.1,  0.2], scale=0.42, size=(45, 2))
lk_sc = rng.normal(loc=[ 2.0, -0.2], scale=0.42, size=(45, 2))

# DeepSAD: NoFlow tight around origin; Leakage scattered outside margin
nf_ds = rng.normal(loc=[0.0, 0.0], scale=0.30, size=(45, 2))
ang   = rng.uniform(0, 2 * np.pi, 40)
rad   = rng.uniform(1.55, 2.30, 40)
lk_ds = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                         gridspec_kw={'wspace': 0.38})

for ax in axes:
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Latent dimension 1', fontsize=11)
    ax.set_ylabel('Latent dimension 2', fontsize=11)
    ax.tick_params(labelsize=9)


# ════════════════════════════════════════════════════════════════════════════
# Panel 1 — SupCon geometry
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_title('SupCon geometry', fontsize=13, fontweight='bold', pad=10)

# light cluster halos
for cx, cy, col in [(-2.1, 0.2, C_NF), (2.0, -0.2, C_LK)]:
    halo = Ellipse((cx, cy), width=1.9, height=1.9,
                   facecolor=col, alpha=0.08, zorder=1)
    ax.add_patch(halo)

# scatter
ax.scatter(nf_sc[:, 0], nf_sc[:, 1],
           color=C_NF, marker='o', s=32, alpha=0.80,
           label='NoFlow', zorder=3, linewidths=0)
ax.scatter(lk_sc[:, 0], lk_sc[:, 1],
           color=C_LK, marker='^', s=38, alpha=0.80,
           label='Leakage', zorder=3, linewidths=0)

# ── "same class pulled together" arrows inside NoFlow cluster
nf_cen = np.array([-2.1, 0.2])
for pt in nf_sc[[3, 12, 27, 38]]:
    delta = nf_cen - pt
    ax.annotate('', xy=pt + 0.60 * delta, xytext=pt + 0.20 * delta,
                arrowprops=dict(arrowstyle='->', color=C_NF,
                                lw=1.1, connectionstyle='arc3,rad=0.15'))

# label for NoFlow pull
ax.annotate('same class\npulled together',
            xy=(-2.1, 0.2), xytext=(-3.5, 1.55),
            fontsize=9, color=C_NF, ha='center',
            arrowprops=dict(arrowstyle='->', color=C_NF, lw=1.0,
                            connectionstyle='arc3,rad=-0.2'))

# ── "same class pulled together" arrows inside Leakage cluster
lk_cen = np.array([2.0, -0.2])
for pt in lk_sc[[5, 17, 30, 41]]:
    delta = lk_cen - pt
    ax.annotate('', xy=pt + 0.60 * delta, xytext=pt + 0.20 * delta,
                arrowprops=dict(arrowstyle='->', color=C_LK,
                                lw=1.1, connectionstyle='arc3,rad=-0.15'))

# ── double-headed arrow between cluster centres
ax.annotate('', xy=(2.0, -0.2), xytext=(-2.1, 0.2),
            arrowprops=dict(arrowstyle='<->', color='#333333',
                            lw=1.6, mutation_scale=14))
mid = np.array([(-2.1 + 2.0) / 2, (0.2 + -0.2) / 2])
ax.text(mid[0] + 0.05, mid[1] + 0.40,
        'different classes\npushed apart',
        fontsize=9, color='#333333', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='none', alpha=0.85))

ax.set_xlim(-4.4, 3.8)
ax.set_ylim(-2.2, 2.8)
ax.legend(loc='lower right', fontsize=9, framealpha=0.85,
          markerscale=1.4, handletextpad=0.5)


# ════════════════════════════════════════════════════════════════════════════
# Panel 2 — DeepSAD geometry
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_title('DeepSAD geometry', fontsize=13, fontweight='bold', pad=10)

MARGIN = 1.20

# margin circle
margin_circle = Circle((0, 0), MARGIN,
                        linestyle='--', linewidth=1.6,
                        edgecolor=C_MRG, facecolor='none', zorder=2)
ax.add_patch(margin_circle)
ax.text(MARGIN * np.cos(np.pi / 4) + 0.07,
        MARGIN * np.sin(np.pi / 4) + 0.10,
        'margin  $m$', fontsize=9, color=C_MRG, ha='left')

# light halo for NoFlow zone
nf_halo = Circle((0, 0), 0.72,
                 facecolor=C_NF, alpha=0.10, zorder=1)
ax.add_patch(nf_halo)

# scatter
ax.scatter(nf_ds[:, 0], nf_ds[:, 1],
           color=C_NF, marker='o', s=32, alpha=0.82,
           label='NoFlow', zorder=3, linewidths=0)
ax.scatter(lk_ds[:, 0], lk_ds[:, 1],
           color=C_LK, marker='^', s=38, alpha=0.82,
           label='Leakage', zorder=3, linewidths=0)

# centre point
ax.scatter([0], [0], color=C_CTR, marker='*', s=220,
           zorder=6, label='Centre  $c$')
ax.text(0.10, -0.22, '$c$', fontsize=12, color=C_CTR,
        fontweight='bold', ha='left')

# ── arrows from NoFlow points toward centre
for pt in nf_ds[[2, 14, 28, 38, 6]]:
    norm = np.linalg.norm(pt)
    if norm < 0.08:
        continue
    tip = pt * 0.18
    tail = pt * 0.80
    ax.annotate('', xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle='->', color=C_NF,
                                lw=1.0, connectionstyle='arc3,rad=0.10'))

# label for NoFlow attraction
ax.annotate('NoFlow attracted\nto centre',
            xy=(0.05, 0.05), xytext=(-2.5, 1.30),
            fontsize=9, color=C_NF, ha='center',
            arrowprops=dict(arrowstyle='->', color=C_NF, lw=1.0,
                            connectionstyle='arc3,rad=0.25'))

# ── arrows from leakage points outward
for pt in lk_ds[[0, 8, 18, 30, 37]]:
    norm = np.linalg.norm(pt)
    direction = pt / norm
    tail = pt * 0.88
    tip  = pt * 1.18
    ax.annotate('', xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle='->', color=C_LK,
                                lw=1.0, connectionstyle='arc3,rad=0.0'))

# label for Leakage repulsion
ax.annotate('Leakage repelled\noutside margin',
            xy=(1.70, -1.70), xytext=(1.80, -2.80),
            fontsize=9, color=C_LK, ha='center',
            arrowprops=dict(arrowstyle='->', color=C_LK, lw=1.0,
                            connectionstyle='arc3,rad=-0.15'))

ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-3.4, 2.8)
ax.legend(loc='upper right', fontsize=9, framealpha=0.85,
          markerscale=1.4, handletextpad=0.5)


# ── save ──────────────────────────────────────────────────────────────────────
if os.getenv('DOCKER_ENV'):
    out_path = Path('/app/images/thesis/supcon_deepsad_geometry.pdf')
else:
    out_path = Path(__file__).resolve().parent.parent / 'figures' / 'supcon_deepsad_geometry.pdf'

out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
print(f'Saved → {out_path}')
plt.close(fig)
