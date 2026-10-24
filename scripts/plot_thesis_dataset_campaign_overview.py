#!/usr/bin/env python
"""
Thesis figure — measurement campaign overview (Chapter 3).

Dataset-level summary only: which recording series exist, when they were
recorded, which house they belong to, and what label availability they
have. NOT a drift or performance plot — no AUC, no GMM scores, no trend
line. Series metadata is taken from the "Available Measurement Series"
table in CLAUDE.md.

Outputs:
  figures/dataset_campaign_overview.pdf
  figures/dataset_campaign_overview.png

Run: python scripts/plot_thesis_dataset_campaign_overview.py
"""
import os
import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ─── colours (fixed categorical order: House A, House B) ─────────────────────
C_HOUSE_A = '#2C6E91'   # muted steel blue
C_HOUSE_B = '#D97642'   # muted warm orange
C_NEUTRAL = '#6e6e6e'   # marker-style legend + excluded series

# ─── label-availability categories ────────────────────────────────────────────
BOTH        = 'both'          # NoFlow + leakage available -> circle
NOFLOW_ONLY = 'noflow_only'   # NoFlow only, AUC undefined -> square
BLIND       = 'blind'         # blind / unlabelled          -> triangle
EXCLUDED    = 'excluded'      # estimated labels, dropped   -> cross

MARKER = {BOTH: 'o', NOFLOW_ONLY: 's', BLIND: '^', EXCLUDED: 'x'}
STYLE_LABEL = {
    BOTH: 'NoFlow + leakage',
    NOFLOW_ONLY: 'NoFlow-only (AUC undefined)',
    BLIND: 'Blind / unlabelled',
    EXCLUDED: 'Excluded (estimated labels)',
}

# ─── series data: (series_id, date, house, category) ──────────────────────────
# Dates parsed from the series name itself (DD.MM.YYYY), matching the
# chronological ordering used throughout the project documentation.
SERIES = [
    # House A
    ('10.01.2026',    datetime.date(2026, 1, 10), 'A', BOTH),
    ('24.01.2026',    datetime.date(2026, 1, 24), 'A', BOTH),
    ('7.02.2026',     datetime.date(2026, 2, 7),  'A', BOTH),
    ('21.02.2026',    datetime.date(2026, 2, 21), 'A', BOTH),
    ('7.03.2026',     datetime.date(2026, 3, 7),  'A', BOTH),
    ('15.03.2026',    datetime.date(2026, 3, 15), 'A', BOTH),
    ('30.03.2026',    datetime.date(2026, 3, 30), 'A', EXCLUDED),
    ('5.04.2026',     datetime.date(2026, 4, 5),  'A', BOTH),
    ('11.04.2026',    datetime.date(2026, 4, 11), 'A', BOTH),
    ('18.04.2026',    datetime.date(2026, 4, 18), 'A', BOTH),
    ('ha-25.04.2026', datetime.date(2026, 4, 25), 'A', BOTH),
    ('ha-2.05.2026',  datetime.date(2026, 5, 2),  'A', BOTH),
    ('ha-9.05.2026',  datetime.date(2026, 5, 9),  'A', NOFLOW_ONLY),
    ('ha-17.05.2026', datetime.date(2026, 5, 17), 'A', BOTH),
    ('ha-24.05.2026', datetime.date(2026, 5, 24), 'A', NOFLOW_ONLY),
    ('ha-2.06.2026',  datetime.date(2026, 6, 2),  'A', BLIND),
    ('ha-6.06.2026',  datetime.date(2026, 6, 6),  'A', BOTH),
    # House B
    ('hb-26.04.2026', datetime.date(2026, 4, 26), 'B', NOFLOW_ONLY),
    ('hb-4.05.2026',  datetime.date(2026, 5, 4),  'B', NOFLOW_ONLY),
    ('hb-10.05.2026', datetime.date(2026, 5, 10), 'B', NOFLOW_ONLY),
    ('hb-17.05.2026', datetime.date(2026, 5, 17), 'B', BOTH),
    ('hb-25.05.2026', datetime.date(2026, 5, 25), 'B', BOTH),
    ('hb-31.05.2026', datetime.date(2026, 5, 31), 'B', BOTH),
    ('hb-7.06.2026',  datetime.date(2026, 6, 7),  'B', NOFLOW_ONLY),
]

Y = {'A': 1.0, 'B': 0.0}
HOUSE_COLOR = {'A': C_HOUSE_A, 'B': C_HOUSE_B}
HOUSE_LABEL = {'A': 'House A', 'B': 'House B'}


def main():
    plt.rcParams.update({
        'font.size': 10.5,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#1a1a1a',
        'text.color': '#1a1a1a',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })

    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor='white')
    ax.set_facecolor('white')

    # baselines for the two site rows
    for house, y in Y.items():
        ax.axhline(y, color='#dddddd', linewidth=1.0, zorder=0)

    for series_id, date, house, cat in SERIES:
        x = mdates.date2num(date)
        y = Y[house]
        color = HOUSE_COLOR[house]
        marker = MARKER[cat]

        if cat == EXCLUDED:
            ax.scatter([x], [y], marker=marker, s=110, color=C_NEUTRAL,
                       linewidths=2.2, zorder=4)
            ax.annotate('excluded\n(estimated labels)', xy=(x, y),
                        xytext=(0, -22), textcoords='offset points',
                        ha='center', va='top', fontsize=7.6,
                        color=C_NEUTRAL, style='italic')
        else:
            face = color if cat != NOFLOW_ONLY else 'white'
            ax.scatter([x], [y], marker=marker, s=85,
                       facecolors=face, edgecolors=color,
                       linewidths=1.4, zorder=4)

    # ─── axes cosmetics ────────────────────────────────────────────────────
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(['House B', 'House A'], fontsize=11)
    ax.set_ylim(-0.6, 1.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(axis='x', which='major', linestyle='-', color='#eeeeee', linewidth=0.9, zorder=0)
    ax.grid(axis='x', which='minor', linestyle='-', color='#f5f5f5', linewidth=0.6, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=9.5)

    ax.set_xlabel('Recording date', fontsize=11, labelpad=8)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0)

    x_all = [mdates.date2num(d) for _, d, _, _ in SERIES]
    pad = (max(x_all) - min(x_all)) * 0.03
    ax.set_xlim(min(x_all) - pad, max(x_all) + pad)

    # ─── legends: site (colour) + label availability (shape) ─────────────
    site_handles = [
        Line2D([0], [0], marker='o', linestyle='none', markersize=8,
               markerfacecolor=HOUSE_COLOR[h], markeredgecolor=HOUSE_COLOR[h],
               label=HOUSE_LABEL[h])
        for h in ('A', 'B')
    ]
    style_handles = [
        Line2D([0], [0], marker=MARKER[cat], linestyle='none', markersize=8,
               markerfacecolor=(C_NEUTRAL if cat != NOFLOW_ONLY else 'white'),
               markeredgecolor=C_NEUTRAL, markeredgewidth=1.6,
               label=STYLE_LABEL[cat])
        for cat in (BOTH, NOFLOW_ONLY, BLIND, EXCLUDED)
    ]

    legend1 = ax.legend(handles=site_handles, title='Installation site',
                         loc='upper left', bbox_to_anchor=(1.01, 1.0),
                         frameon=False, fontsize=9.5, title_fontsize=10,
                         handletextpad=0.6, borderaxespad=0.0)
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, title='Label availability',
              loc='upper left', bbox_to_anchor=(1.01, 0.62),
              frameon=False, fontsize=9.5, title_fontsize=10,
              handletextpad=0.6, borderaxespad=0.0)

    fig.tight_layout(rect=(0, 0, 0.80, 1))

    if os.getenv('DOCKER_ENV'):
        out_dir = Path('/app/figures')
    else:
        out_dir = Path(__file__).resolve().parent.parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext, dpi in [('pdf', None), ('png', 300)]:
        p = out_dir / f'dataset_campaign_overview.{ext}'
        kw = dict(format=ext, bbox_inches='tight', facecolor='white')
        if dpi:
            kw['dpi'] = dpi
        fig.savefig(p, **kw)
        print(f'Saved → {p}')

    plt.close(fig)


if __name__ == '__main__':
    main()
