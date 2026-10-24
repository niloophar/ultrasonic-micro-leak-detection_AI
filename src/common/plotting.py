"""
Common plotting functions for preprocessing pipeline.

This module consolidates plotting functions for:
- Label Data: Series overview plots
- Clean Data: Outlier visualization
- Extract Relaxations: Block-level analysis
- Thermal Equilibrium: Equilibrium region visualization
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('agg')  # Non-interactive backend for saving plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_leakage_times(
    df: pd.DataFrame,
    leakage_rule: dict
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Get leakage start and end times from flow_type column.

    Args:
        df: DataFrame with 'Time' and 'flow_type' columns.
        leakage_rule: Leakage rule from config (for reference, not used directly).

    Returns:
        Tuple of (leakage_start, leakage_end) as Timestamps, or (None, None) if no leakage.
    """
    if 'flow_type' not in df.columns:
        return None, None

    flow_mask = df['flow_type'] == 'flow'
    if not flow_mask.any():
        return None, None

    flow_times = df.loc[flow_mask, 'Time']
    leakage_start = flow_times.min()
    leakage_end = flow_times.max()

    return leakage_start, leakage_end


def _add_leakage_lines(
    ax,
    leakage_start: Optional[pd.Timestamp],
    leakage_end: Optional[pd.Timestamp]
):
    """
    Add vertical lines for leakage start and end to a plot.

    Args:
        ax: Matplotlib axes object.
        leakage_start: Start timestamp of leakage period.
        leakage_end: End timestamp of leakage period.
    """
    if leakage_start is not None:
        ax.axvline(x=leakage_start, color='green', linestyle='--', linewidth=2,
                   label=f'Leakage Start ({leakage_start.strftime("%H:%M")})')
    if leakage_end is not None:
        ax.axvline(x=leakage_end, color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Leakage End ({leakage_end.strftime("%H:%M")})')


# =============================================================================
# LABEL DATA PLOTS
# =============================================================================

def plot_series(
    df: pd.DataFrame,
    title: str,
    series_name: str,
    img_path: Path,
    y_col: str = 't_up_0 in ps',
    N: int = 64
):
    """
    Plot y_col over time per flow_type after averaging in blocks of size N.

    Args:
        df: DataFrame with 'Time', 'flow_type', 'sample' columns.
        title: Plot title.
        series_name: Series name for save path.
        img_path: Base path for saving images.
        y_col: Y-axis column name.
        N: Block size for averaging.
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()

    df_block = (
        df.sort_values('Time')
          .assign(block=lambda d: d.groupby(['flow_type', 'sample']).cumcount() // N)
          .groupby(['flow_type', 'sample', 'block'], as_index=False)
          .agg(**{'Time': ('Time', 'mean')}, **{c: (c, 'mean') for c in num_cols})
          .drop(columns='block')
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f'{title} (Averaged N={N})', fontsize=16)

    for flow_type, grp in df_block.groupby('flow_type'):
        num_points = len(grp[y_col])
        ax.plot(grp['Time'], grp[y_col], label=f'{flow_type} (num: {num_points})',
                alpha=0.7, marker='.', ms=2, lw=0)

    ax.set_xlabel('Timestamp', fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.grid()
    ax.legend()

    plt.tight_layout()

    target_path = img_path / 'label' / series_name / f'{title}.png'
    target_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(target_path)
    plt.close('all')


# =============================================================================
# CLEAN DATA PLOTS
# =============================================================================


def plot_temperature_combined(
    df: pd.DataFrame,
    title: str,
    plot_func: str,
    leakage_times: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None),
    block_size: int = 64,
    save_path: Optional[Path] = None,
):
    """
    Combined temperature plot with 2 subplots.

    Top: Temp Pipe and Temp Ambient overlaid.
    Bottom: Temp Diff (Pipe - Ambient).

    Args:
        df: DataFrame with valid column and temperature columns.
        title: Plot title.
        plot_func: Which outlier view to use: "all", "local", "inliers", "inliers_leakage".
        leakage_times: Tuple of (start, end) timestamps.
        block_size: Block size for averaging (used in inliers views).
        save_path: Optional path to save the plot.
    """
    pipe_col = "Temp Pipe in °C"
    ambient_col = "Temp Ambient in °C"
    if pipe_col not in df.columns or ambient_col not in df.columns:
        return

    # Select data subset based on plot_func
    if plot_func == "all":
        df_plot = df
        subtitle = "All Points"
    elif plot_func == "local":
        df_plot = df[df['valid'] != -1]
        subtitle = "Local Outliers (no global)"
    elif plot_func == "inliers":
        df_plot = df[df['valid'] == 1].copy()
        subtitle = f"Inliers Only (averaged N={block_size})"
    elif plot_func == "inliers_leakage":
        ls, le = leakage_times
        if ls is None or le is None:
            return
        df_plot = df[(df['Time'] >= ls) & (df['Time'] <= le)]
        df_plot = df_plot[df_plot['valid'] == 1]
        subtitle = "Inliers in Leakage Window"
    else:
        return

    if df_plot.empty:
        return

    # Compute diff
    temp_diff = df_plot[pipe_col] - df_plot[ambient_col]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{title} - {subtitle}", fontsize=14)

    # Determine valid categories for coloring
    if plot_func == "all":
        _plot_temp_by_validity(ax1, df_plot, pipe_col, ambient_col, leakage_times)
        _plot_temp_diff_by_validity(ax2, df_plot, temp_diff, leakage_times)
    elif plot_func == "local":
        _plot_temp_by_validity(ax1, df_plot, pipe_col, ambient_col, leakage_times,
                               show_global=False)
        _plot_temp_diff_by_validity(ax2, df_plot, temp_diff, leakage_times,
                                    show_global=False)
    elif plot_func == "inliers":
        # Average in blocks
        df_sorted = df_plot.sort_values('Time').reset_index(drop=True)
        df_sorted['_block'] = df_sorted.index // block_size
        averaged = df_sorted.groupby('_block', as_index=False).agg(
            Time=('Time', 'mean'),
            **{pipe_col: (pipe_col, 'mean'),
               ambient_col: (ambient_col, 'mean')}
        )
        averaged['_diff'] = averaged[pipe_col] - averaged[ambient_col]
        n_pts = len(df_sorted)
        n_blk = len(averaged)

        ax1.scatter(averaged['Time'], averaged[pipe_col], s=8, alpha=0.7,
                    color='red', label=f'Pipe ({n_pts:,} pts → {n_blk:,} blocks)')
        ax1.scatter(averaged['Time'], averaged[ambient_col], s=8, alpha=0.7,
                    color='blue', label=f'Ambient')
        _add_leakage_lines(ax1, leakage_times[0], leakage_times[1])

        ax2.scatter(averaged['Time'], averaged['_diff'], s=8, alpha=0.7,
                    color='purple')
        _add_leakage_lines(ax2, leakage_times[0], leakage_times[1])
    elif plot_func == "inliers_leakage":
        ax1.scatter(df_plot['Time'], df_plot[pipe_col], s=6, alpha=0.6,
                    color='red', label=f'Pipe ({len(df_plot):,})')
        ax1.scatter(df_plot['Time'], df_plot[ambient_col], s=6, alpha=0.6,
                    color='blue', label=f'Ambient')
        ls, le = leakage_times
        _add_leakage_lines(ax1, ls, le)
        try:
            dt = le - ls
            pad = dt * 0.02
            if pad == pd.Timedelta(0):
                pad = pd.Timedelta(seconds=1)
            ax1.set_xlim(ls - pad, le + pad)
        except Exception:
            pass

        ax2.scatter(df_plot['Time'], temp_diff, s=6, alpha=0.6, color='purple')
        _add_leakage_lines(ax2, ls, le)
        try:
            ax2.set_xlim(ls - pad, le + pad)
        except Exception:
            pass

    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Temp Diff (Pipe−Ambient) (°C)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close('all')


def _plot_temp_by_validity(ax, df, pipe_col, ambient_col, leakage_times,
                           show_global=True):
    """Helper: scatter pipe and ambient temps colored by validity."""
    inliers = df[df['valid'] == 1]
    local_out = df[df['valid'] == 0]

    if len(inliers) > 0:
        ax.scatter(inliers['Time'], inliers[pipe_col], s=2, alpha=0.5,
                   color='red', label=f'Pipe inlier ({len(inliers):,})')
        ax.scatter(inliers['Time'], inliers[ambient_col], s=2, alpha=0.5,
                   color='blue', label=f'Ambient inlier')
    if len(local_out) > 0:
        ax.scatter(local_out['Time'], local_out[pipe_col], s=10, alpha=0.6,
                   color='orange', marker='x', label=f'Local outlier ({len(local_out):,})')
    if show_global:
        global_out = df[df['valid'] == -1]
        if len(global_out) > 0:
            ax.scatter(global_out['Time'], global_out[pipe_col], s=20, alpha=0.6,
                       color='darkred', marker='x',
                       label=f'Global outlier ({len(global_out):,})')

    _add_leakage_lines(ax, leakage_times[0], leakage_times[1])


def _plot_temp_diff_by_validity(ax, df, temp_diff, leakage_times, show_global=True):
    """Helper: scatter temp diff colored by validity."""
    inliers_mask = df['valid'] == 1
    local_mask = df['valid'] == 0

    if inliers_mask.any():
        ax.scatter(df.loc[inliers_mask, 'Time'], temp_diff[inliers_mask],
                   s=2, alpha=0.5, color='purple')
    if local_mask.any():
        ax.scatter(df.loc[local_mask, 'Time'], temp_diff[local_mask],
                   s=10, alpha=0.6, color='orange', marker='x')
    if show_global:
        global_mask = df['valid'] == -1
        if global_mask.any():
            ax.scatter(df.loc[global_mask, 'Time'], temp_diff[global_mask],
                       s=20, alpha=0.6, color='darkred', marker='x')

    _add_leakage_lines(ax, leakage_times[0], leakage_times[1])


def plot_all_outliers(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    leakage_times: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None),
    save_path: Optional[Path] = None
):
    """
    Plot all points - Inliers (blue), local outliers (orange), global outliers (red).

    Args:
        df: DataFrame with 'valid' column.
        title: Plot title.
        value_col: Y-axis column.
        leakage_times: Tuple of (start, end) timestamps for leakage period.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"{title} - All Points", fontsize=16)

    # Inliers (valid == 1) - BLUE
    inliers = df[df['valid'] == 1]
    ax.scatter(inliers['Time'], inliers[value_col],
               marker='o', s=2, color='blue', alpha=0.5,
               label=f'Inliers ({len(inliers):,})')

    # Local outliers (valid == 0) - ORANGE
    local_out = df[df['valid'] == 0]
    ax.scatter(local_out['Time'], local_out[value_col],
               marker='x', s=10, color='orange',
               label=f'Local Outliers ({len(local_out):,})')

    # Global outliers (valid == -1) - RED
    global_out = df[df['valid'] == -1]
    ax.scatter(global_out['Time'], global_out[value_col],
               marker='x', s=20, color='red',
               label=f'Global Outliers ({len(global_out):,})')

    _add_leakage_lines(ax, leakage_times[0], leakage_times[1])

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        
    plt.close('all')


def plot_local_outliers(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    leakage_times: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None),
    save_path: Optional[Path] = None
):
    """
    Plot local outliers + inliers only (no global outliers).

    Args:
        df: DataFrame with 'valid' column.
        title: Plot title.
        value_col: Y-axis column.
        leakage_times: Tuple of (start, end) timestamps for leakage period.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"{title} - Local Outliers (no global)", fontsize=16)

    df_no_global = df[df['valid'] != -1]

    inliers = df_no_global[df_no_global['valid'] == 1]
    ax.scatter(inliers['Time'], inliers[value_col],
               marker='o', s=2, color='blue', alpha=0.5,
               label=f'Inliers ({len(inliers):,})')

    local_out = df_no_global[df_no_global['valid'] == 0]
    ax.scatter(local_out['Time'], local_out[value_col],
               marker='x', s=10, color='orange',
               label=f'Local Outliers ({len(local_out):,})')

    _add_leakage_lines(ax, leakage_times[0], leakage_times[1])

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        
    plt.close('all')


def plot_inliers_averaged(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    block_size: int = 64,
    leakage_times: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None),
    save_path: Optional[Path] = None
):
    """
    Plot inliers only, averaged over blocks of N points.

    Args:
        df: DataFrame with 'valid' column.
        title: Plot title.
        value_col: Y-axis column.
        block_size: Number of points to average.
        leakage_times: Tuple of (start, end) timestamps for leakage period.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"{title} - Inliers Only (averaged N={block_size})", fontsize=16)

    inliers = df[df['valid'] == 1].copy()

    if inliers.empty:
        ax.text(0.5, 0.5, 'No inliers found', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    else:
        inliers = inliers.sort_values('Time').reset_index(drop=True)
        inliers['_block'] = inliers.index // block_size

        averaged = inliers.groupby('_block', as_index=False).agg(
            Time=('Time', 'mean'),
            **{value_col: (value_col, 'mean')}
        )

        ax.scatter(averaged['Time'], averaged[value_col],
                   marker='o', s=8, color='blue', alpha=0.7,
                   label=f'Inliers ({len(inliers):,} pts -> {len(averaged):,} blocks)')

    _add_leakage_lines(ax, leakage_times[0], leakage_times[1])

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        
    plt.close('all')


def plot_inliers_leakage(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    block_size: int = 64,
    leakage_times: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None),
    save_path: Optional[Path] = None
):
    """
    Plot inliers within leakage window only (no averaging).

    Args:
        df: DataFrame with 'valid' column and 'Time' column.
        title: Plot title.
        value_col: Y-axis column.
        block_size: Unused (kept for API compatibility).
        leakage_times: Tuple of (start, end) timestamps for leakage period.
        save_path: Optional path to save the plot.
    """
    leakage_start, leakage_end = leakage_times

    if leakage_start is None or leakage_end is None:
                return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"{title} - Inliers in Leakage Window", fontsize=16)

    df_leak = df[(df['Time'] >= leakage_start) & (df['Time'] <= leakage_end)]
    inliers = df_leak[df_leak['valid'] == 1]

    if inliers.empty:
        ax.text(0.5, 0.5, 'No inliers in leakage window', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    else:
        ax.scatter(inliers['Time'], inliers[value_col],
                   marker='o', s=6, color='blue', alpha=0.6,
                   label=f'Inliers in leakage ({len(inliers):,})')

    _add_leakage_lines(ax, leakage_start, leakage_end)

    try:
        dt = leakage_end - leakage_start
        pad = dt * 0.02
        if pad == pd.Timedelta(0):
            pad = pd.Timedelta(seconds=1)
        ax.set_xlim(leakage_start - pad, leakage_end + pad)
    except Exception:
        pass

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        
    plt.close('all')


# =============================================================================
# RELAXATION PLOTS
# =============================================================================

def plot_relaxations_overview(
    df_blocks: pd.DataFrame,
    jump_block_indices: np.ndarray,
    metadata_list: List[dict],
    title: str,
    value_col: str,
    save_path: Optional[Path] = None,
    leakage_times: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None)
):
    """
    Plot relaxations overview: block means with jumps and color-coded relaxations.

    Creates a 2-subplot figure:
    1. Block overview with jump markers
    2. Relaxations color-coded

    Args:
        df_blocks: Block-level DataFrame with relaxation_id column.
        jump_block_indices: Array of block indices where jumps occur.
        metadata_list: List of relaxation metadata dicts.
        title: Plot title.
        value_col: Y-axis column label.
        save_path: Optional path to save the plot.
        leakage_times: Tuple of (start, end) timestamps for leakage period.
    """
    if len(df_blocks) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{title}", fontsize=14)

    colors = plt.cm.tab10.colors
    leakage_start, leakage_end = leakage_times

    # --- Subplot 1: Block Overview with Jumps ---
    ax1 = axes[0]
    ax1.set_title("Block Means with Jump Detection", fontsize=12)

    # Plot all blocks
    ax1.scatter(df_blocks['block_time'], df_blocks['block_mean'],
                s=10, alpha=0.7, color='blue', label=f'Blocks ({len(df_blocks)})')

    # Mark jumps
    for jump_idx in jump_block_indices:
        if jump_idx < len(df_blocks):
            jump_time = df_blocks['block_time'].iloc[jump_idx]
            ax1.axvline(x=jump_time, color='red', linestyle='--',
                        linewidth=1.5, alpha=0.7)

    # Add leakage lines to subplot 1
    _add_leakage_lines(ax1, leakage_start, leakage_end)

    ax1.set_ylabel(value_col, fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Relaxations Color-Coded ---
    ax2 = axes[1]
    ax2.set_title(f"Extracted Relaxations ({len(metadata_list)} found)", fontsize=12)

    for i, meta in enumerate(metadata_list):
        rel_id = meta['relaxation_id']
        start_block = meta['start_block']
        end_block = meta['end_block']

        segment = df_blocks.iloc[start_block:end_block]
        color = colors[i % len(colors)]

        label = f"R{rel_id} ({meta['n_blocks']} blocks)" if i < 10 else None
        ax2.scatter(segment['block_time'], segment['block_mean'],
                    s=15, color=color, alpha=0.8, label=label)

    # Add leakage lines to subplot 2
    _add_leakage_lines(ax2, leakage_start, leakage_end)

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel(value_col, fontsize=11)
    if len(metadata_list) <= 10:
        ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close('all')


# =============================================================================
# EQUILIBRIUM PLOTS
# =============================================================================


def plot_equilibrium_detection(
    df_blocks: pd.DataFrame,
    equilibrium_mask: np.ndarray,
    slopes: np.ndarray,
    eq_config: dict,
    title: str,
    value_col: str,
    save_path: Optional[Path] = None,
    leakage_windows: Optional[List[Tuple]] = None,
    slope_times: Optional[List] = None,
):
    """
    Plot equilibrium detection results.

    Creates a 2-subplot figure:
    1. Top: Block means with equilibrium blocks highlighted in green
    2. Bottom: Slope per window with threshold lines

    Args:
        df_blocks: Block-level DataFrame with block_time and block_mean.
        equilibrium_mask: Boolean array (True = equilibrium block).
        slopes: Array of slopes per window position.
        eq_config: equilibrium_config section (for threshold/window params).
        title: Plot title.
        value_col: Y-axis column label.
        save_path: Optional path to save the plot.
        leakage_windows: Optional list of (start, end) tuples for shading.
        slope_times: Optional pre-computed timestamps for each slope value.
            Used when slopes come from multiple separate segments (e.g. relaxations).
    """
    if len(df_blocks) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"Equilibrium Detection - {title}", fontsize=14)

    times = pd.to_datetime(df_blocks['block_time'])
    means = df_blocks['block_mean'].values

    # --- Subplot 1: Block means with equilibrium highlighting ---
    ax1 = axes[0]
    ax1.set_title("Block Means with Equilibrium Regions", fontsize=12)

    # Non-equilibrium blocks in blue
    non_eq = ~equilibrium_mask[:len(means)]
    if non_eq.any():
        ax1.scatter(
            times[non_eq], means[non_eq],
            s=8, alpha=0.5, color='#6495ED',
            label=f'Non-equilibrium ({int(non_eq.sum())})',
        )

    # Equilibrium blocks in pastel green
    eq_mask = equilibrium_mask[:len(means)]
    if eq_mask.any():
        ax1.scatter(
            times[eq_mask], means[eq_mask],
            s=10, alpha=0.8, color='#A8D5BA',
            label=f'Equilibrium ({int(eq_mask.sum())})',
        )

    # Shade leakage windows
    if leakage_windows:
        for i, (win_start, win_end) in enumerate(leakage_windows):
            label = 'Leakage window' if i == 0 else None
            ax1.axvspan(win_start, win_end, alpha=0.1, color='red', label=label)

    ax1.set_ylabel(value_col, fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Slopes per window ---
    ax2 = axes[1]
    slope_threshold = eq_config.get('slope_threshold', 0.5)

    if len(slopes) > 0:
        if slope_times is not None and len(slope_times) == len(slopes):
            # Pre-computed slope times (e.g. concatenated from multiple relaxations)
            ax2.scatter(slope_times, slopes, s=4, alpha=0.7, color='blue', label='Window slope')
        else:
            window_length = eq_config.get('window_length', 100)
            hop_size = eq_config.get('hop_size', 50)

            # Compute window center times
            window_indices = list(range(0, len(means) - window_length + 1, hop_size))
            if len(window_indices) == len(slopes):
                window_centers = [
                    times.iloc[wi + window_length // 2] if (wi + window_length // 2) < len(times) else times.iloc[-1]
                    for wi in window_indices
                ]
                ax2.plot(window_centers, slopes, 'b-', lw=1, alpha=0.7, label='Window slope')
            else:
                ax2.plot(slopes, 'b-', lw=1, alpha=0.7, label='Window slope')
        ax2.set_title(f"Slope per Window (threshold = +/-{slope_threshold} ps/s)", fontsize=12)
    else:
        ax2.set_title("Slopes (not available)", fontsize=12)

    ax2.axhline(slope_threshold, color='#D5A6E6', linestyle='--', lw=1.5,
                label=f'+{slope_threshold}')
    ax2.axhline(-slope_threshold, color='#D5A6E6', linestyle='--', lw=1.5,
                label=f'-{slope_threshold}')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Slope (ps/s)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close('all')


def plot_equilibrium_temperature(
    df: pd.DataFrame,
    time_col: str,
    title: str,
    save_path: Optional[Path] = None,
):
    """
    Plot temperature for equilibrium regions only.

    Uses the 'equilibrium' column already set on df to filter data.
    Only shows equilibrium points — no context, no leakage shading.

    Creates a 2-subplot figure:
    Top: Temp Pipe and Temp Ambient for equilibrium points only.
    Bottom: Temp Diff (Pipe - Ambient).

    Args:
        df: Full DataFrame with temperature columns, 'equilibrium', and time_col.
        time_col: Time column name.
        title: Plot title.
        save_path: Optional path to save the plot.
    """
    pipe_col = "Temp Pipe in °C"
    ambient_col = "Temp Ambient in °C"
    if pipe_col not in df.columns or ambient_col not in df.columns:
        return
    if 'equilibrium' not in df.columns:
        return

    df_eq = df[df['equilibrium'] == 1].sort_values(time_col)
    if df_eq.empty:
        return

    times = df_eq[time_col]
    temp_diff = df_eq[pipe_col].values - df_eq[ambient_col].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f"Temperature (Equilibrium) - {title}", fontsize=14)

    # Top: Pipe and Ambient
    ax1.scatter(times, df_eq[pipe_col].values,
                s=6, alpha=0.7, color='red', label='Pipe')
    ax1.scatter(times, df_eq[ambient_col].values,
                s=6, alpha=0.7, color='blue', label='Ambient')

    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: Diff
    ax2.scatter(times, temp_diff,
                s=6, alpha=0.7, color='purple', label='Pipe − Ambient')

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Temp Diff (Pipe−Ambient) (°C)', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close('all')


# =============================================================================
# DRIFT OVERVIEW PLOT
# =============================================================================

_AMB_TEMP_COL = "Temp Ambient in °C"


def _is_house_b(series_name: str) -> bool:
    return series_name.startswith('hb-')


def _series_marker(series_name: str) -> str:
    return '^' if _is_house_b(series_name) else 'D'


def _fit_poly_for_house(times: list, values: list, fit_degree: int):
    """Return (t_fit list, y_fit array) for a polynomial fit, or (None, None) if < 2 points."""
    if len(times) < 2:
        return None, None
    t0 = min(times)
    x_num = np.array([(t - t0).total_seconds() for t in times])
    y_arr = np.array(values, dtype=float)
    sort_idx = np.argsort(x_num)
    deg = min(fit_degree, len(times) - 1)
    coeffs = np.polyfit(x_num[sort_idx], y_arr[sort_idx], deg)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(x_num.min(), x_num.max(), 200)
    t_fit = [t0 + pd.Timedelta(seconds=float(s)) for s in x_fit]
    return t_fit, poly(x_fit)


def _attach_outside_legend(ax, handles, labels):
    """Place legend above the axes in multiple columns with enlarged markers."""
    ncol = max(1, (len(handles) + 1) // 2)
    ax.legend(
        handles, labels,
        loc='lower center', bbox_to_anchor=(0.5, 1.01),
        ncol=ncol, fontsize=9,
        framealpha=0.9, borderaxespad=0.0,
    )


def _plot_drift_on_ax(ax, series_data, value_col, fit_degree, colors):
    """
    Shared rendering logic for a single drift-overview axes (series means variant).
    Returns (legend_handles, legend_labels).
    """
    mean_times_ha: list = []
    mean_values_ha: list = []
    mean_times_hb: list = []
    mean_values_hb: list = []
    legend_handles = []
    legend_labels = []

    for i, sd in enumerate(series_data):
        name = sd['name']
        color = colors[i % len(colors)]
        mkr = _series_marker(name)

        t = pd.to_datetime(sd['times'], format='ISO8601')
        v = np.asarray(sd['values'], dtype=float)

        ax.scatter(t, v, s=4, alpha=0.4, color=color, marker=mkr, edgecolors='none')

        if len(v) > 0:
            t_mean = t.min() + (t.max() - t.min()) / 2
            v_mean = float(np.mean(v))
            h = ax.scatter([t_mean], [v_mean], s=150, color=color, marker=mkr,
                           edgecolors='black', linewidths=1.5, zorder=5, label=name)
            legend_handles.append(h)
            legend_labels.append(name)
            if _is_house_b(name):
                mean_times_hb.append(t_mean)
                mean_values_hb.append(v_mean)
            else:
                mean_times_ha.append(t_mean)
                mean_values_ha.append(v_mean)

    for times, values, fit_color, house_label in [
        (mean_times_ha, mean_values_ha, 'orange', 'House A'),
        (mean_times_hb, mean_values_hb, '#1f77b4', 'House B'),
    ]:
        t_fit, y_fit = _fit_poly_for_house(times, values, fit_degree)
        if t_fit is None:
            continue
        deg = min(fit_degree, len(times) - 1)
        lbl = f'Trend {house_label} (deg={deg})'
        (h,) = ax.plot(t_fit, y_fit, color=fit_color, linewidth=2.5, zorder=4, label=lbl)
        legend_handles.append(h)
        legend_labels.append(lbl)

    return legend_handles, legend_labels


def _plot_drift_files_on_ax(ax, series_data, value_col, fit_degree, colors):
    """
    Shared rendering logic for a single drift-overview axes (per-file means variant).
    Returns (legend_handles, legend_labels).
    """
    all_file_times_ha: list = []
    all_file_values_ha: list = []
    all_file_times_hb: list = []
    all_file_values_hb: list = []
    legend_handles = []
    legend_labels = []

    for i, sd in enumerate(series_data):
        name = sd['name']
        color = colors[i % len(colors)]
        mkr = _series_marker(name)

        t = pd.to_datetime(sd['times'], format='ISO8601')
        v = np.asarray(sd['values'], dtype=float)

        ax.scatter(t, v, s=4, alpha=0.25, color=color, marker=mkr, edgecolors='none')

        files = sd.get('files', [])
        series_in_legend = False
        for fd in files:
            ft = pd.to_datetime(fd['mean_time'])
            fv = fd['mean_value']
            h = ax.scatter([ft], [fv], s=120, color=color, marker=mkr,
                           edgecolors='black', linewidths=1.0, zorder=5)
            if not series_in_legend:
                legend_handles.append(h)
                legend_labels.append(name)
                series_in_legend = True
            if _is_house_b(name):
                all_file_times_hb.append(ft)
                all_file_values_hb.append(fv)
            else:
                all_file_times_ha.append(ft)
                all_file_values_ha.append(fv)

    for times, values, fit_color, house_label in [
        (all_file_times_ha, all_file_values_ha, 'orange', 'House A'),
        (all_file_times_hb, all_file_values_hb, '#1f77b4', 'House B'),
    ]:
        t_fit, y_fit = _fit_poly_for_house(times, values, fit_degree)
        if t_fit is None:
            continue
        deg = min(fit_degree, len(times) - 1)
        lbl = f'Trend {house_label} (deg={deg}, {len(times)} files)'
        (h,) = ax.plot(t_fit, y_fit, color=fit_color, linewidth=2.5, zorder=4, label=lbl)
        legend_handles.append(h)
        legend_labels.append(lbl)

    return legend_handles, legend_labels


def _plot_ambient_temp_on_ax(ax, series_data, colors):
    """Plot block-averaged ambient temperature per series on the given axes."""
    for i, sd in enumerate(series_data):
        name = sd['name']
        color = colors[i % len(colors)]
        mkr = _series_marker(name)

        temp_vals = sd.get('temp_amb_values', [])
        if not temp_vals:
            continue

        t = pd.to_datetime(sd['times'], format='ISO8601')
        v = np.array([x if x is not None else np.nan for x in temp_vals], dtype=float)
        valid = ~np.isnan(v)
        if not valid.any():
            continue

        ax.scatter(t[valid], v[valid], s=4, alpha=0.4, color=color, marker=mkr,
                   edgecolors='none')

        # Per-series mean highlighted
        t_mean = t[valid].min() + (t[valid].max() - t[valid].min()) / 2
        v_mean = float(np.nanmean(v[valid]))
        ax.scatter([t_mean], [v_mean], s=150, color=color, marker=mkr,
                   edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_ylabel(_AMB_TEMP_COL, fontsize=11)
    ax.grid(True, alpha=0.3)


def _plot_ambient_temp_files_on_ax(ax, series_data, colors):
    """Plot per-file mean ambient temperature per series on the given axes."""
    for i, sd in enumerate(series_data):
        name = sd['name']
        color = colors[i % len(colors)]
        mkr = _series_marker(name)

        files = sd.get('files', [])
        for fd in files:
            fv = fd.get('mean_temp_amb')
            if fv is None:
                continue
            ft = pd.to_datetime(fd['mean_time'])
            ax.scatter([ft], [fv], s=120, color=color, marker=mkr,
                       edgecolors='black', linewidths=1.0, zorder=5)

    ax.set_ylabel(_AMB_TEMP_COL, fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_drift_overview(
    series_data: List[dict],
    value_col: str,
    fit_degree: int,
    save_path: Path,
):
    """
    Plot all cleaned series in one figure to visualise long-term drift.

    Each series is shown as a scatter of block-averaged inlier values.
    Per-series means are highlighted with house-specific markers (D = House A,
    ^ = House B) and separate polynomial trend lines are fitted per house.
    Legend is placed above the plot in multiple columns.

    Args:
        series_data: List of dicts with keys 'name', 'times', 'values'.
        value_col: Y-axis label (e.g. 't_up_0 in ps').
        fit_degree: Polynomial degree for the trend curves.
        save_path: Path to save the figure.
    """
    if not series_data:
        return

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)

    handles, labels = _plot_drift_on_ax(ax, series_data, value_col, fit_degree, colors)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_title("Drift Overview - All Series (cleaned inliers)", fontsize=14)
    ax.grid(True, alpha=0.3)
    _attach_outside_legend(ax, handles, labels)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close('all')


def plot_drift_overview_files(
    series_data: List[dict],
    value_col: str,
    fit_degree: int = 3,
    save_path: Optional[Path] = None,
):
    """
    Drift overview with per-file mean markers and per-house polynomial fits.

    Same scatter background as plot_drift_overview, but instead of per-series
    means, each CSV file contributes one marker. House B series (hb- prefix) use
    triangle markers; House A uses diamonds. Fits are computed separately per house.
    Legend is placed above the plot in multiple columns.

    Args:
        series_data: List of dicts with keys 'name', 'times', 'values', 'files'.
        value_col: Y-axis label.
        fit_degree: Polynomial degree for the trend curves.
        save_path: Path to save the figure.
    """
    if not series_data:
        return

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)

    handles, labels = _plot_drift_files_on_ax(ax, series_data, value_col, fit_degree, colors)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_title("Drift Overview - Per-File Means (cleaned inliers)", fontsize=14)
    ax.grid(True, alpha=0.3)
    _attach_outside_legend(ax, handles, labels)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close('all')


def plot_drift_overview_subplots(
    series_data: List[dict],
    value_col: str,
    fit_degree: int,
    save_path: Path,
):
    """
    Two-subplot version of plot_drift_overview.

    Top subplot: drift overview (series means, per-house fits).
    Bottom subplot: block-averaged Ambient Temperature by house.
    Legend is shared above the top subplot.

    Args:
        series_data: List of dicts with keys 'name', 'times', 'values',
            and optionally 'temp_amb_values'.
        value_col: Y-axis label for the top subplot.
        fit_degree: Polynomial degree for the trend curves.
        save_path: Path to save the figure.
    """
    if not series_data:
        return

    colors = plt.cm.tab10.colors
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 11),
                                          constrained_layout=True)

    handles, labels = _plot_drift_on_ax(ax_top, series_data, value_col, fit_degree, colors)

    ax_top.set_xlabel('Time', fontsize=12)
    ax_top.set_ylabel(value_col, fontsize=12)
    ax_top.set_title("Drift Overview - All Series (cleaned inliers)", fontsize=14)
    ax_top.grid(True, alpha=0.3)
    _attach_outside_legend(ax_top, handles, labels)

    _plot_ambient_temp_on_ax(ax_bot, series_data, colors)
    ax_bot.set_xlabel('Time', fontsize=12)
    ax_bot.set_title("Ambient Temperature by House", fontsize=13)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close('all')


def plot_drift_overview_files_subplots(
    series_data: List[dict],
    value_col: str,
    fit_degree: int = 3,
    save_path: Optional[Path] = None,
):
    """
    Two-subplot version of plot_drift_overview_files.

    Top subplot: per-file means with per-house fits.
    Bottom subplot: per-file mean Ambient Temperature by house.
    Legend is shared above the top subplot.

    Args:
        series_data: List of dicts with keys 'name', 'times', 'values', 'files'.
            Each file entry may include 'mean_temp_amb'.
        value_col: Y-axis label for the top subplot.
        fit_degree: Polynomial degree for the trend curves.
        save_path: Path to save the figure.
    """
    if not series_data:
        return

    colors = plt.cm.tab10.colors
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 11),
                                          constrained_layout=True)

    handles, labels = _plot_drift_files_on_ax(ax_top, series_data, value_col, fit_degree, colors)

    ax_top.set_xlabel('Time', fontsize=12)
    ax_top.set_ylabel(value_col, fontsize=12)
    ax_top.set_title("Drift Overview - Per-File Means (cleaned inliers)", fontsize=14)
    ax_top.grid(True, alpha=0.3)
    _attach_outside_legend(ax_top, handles, labels)

    _plot_ambient_temp_files_on_ax(ax_bot, series_data, colors)
    ax_bot.set_xlabel('Time', fontsize=12)
    ax_bot.set_title("Ambient Temperature per File by House", fontsize=13)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close('all')


# =============================================================================
# CLASSIFICATION PLOTS
# =============================================================================

def plot_feature_histogram(
    feature_name: str,
    series_data: dict,
    save_path: Path,
):
    """
    Plot flow vs noflow histograms for a single feature, one subplot per series.

    Args:
        feature_name: Name of the feature being plotted.
        series_data: Mapping of series_name -> {'flow': np.array, 'noflow': np.array}.
        save_path: Path to save the figure.
    """
    n = len(series_data)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (series_name, class_data) in zip(axes, series_data.items()):
        noflow_vals = class_data.get('noflow', np.array([]))
        flow_vals = class_data.get('flow', np.array([]))

        if len(noflow_vals) > 0:
            ax.hist(noflow_vals, bins=200, density=True, histtype='step',
                    color='#6495ED', linewidth=1.2, label=f'noflow ({len(noflow_vals):,})')
        if len(flow_vals) > 0:
            ax.hist(flow_vals, bins=200, density=True, histtype='step',
                    color='#E8A0BF', linewidth=1.2, label=f'flow ({len(flow_vals):,})')

        ax.set_title(series_name, fontsize=12)
        ax.set_xlabel(feature_name, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(feature_name, fontsize=14)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_segment_feature_drift(
    feature_name: str,
    series_data: List[dict],
    fit_degree: int,
    save_path: Path,
):
    """
    Drift overview for one feature: 3 subplots (All, NoFlow, Leakage).

    Each subplot shows per-segment mean values as scatter per series,
    with per-series means highlighted and a polynomial trend line.

    Args:
        feature_name: Column/feature name for the y-axis.
        series_data: List of dicts with keys 'name', and per subset:
            'times_all'/'values_all', 'times_noflow'/'values_noflow',
            'times_flow'/'values_flow'.
        fit_degree: Polynomial degree for trend curve.
        save_path: Path to save the figure.
    """
    if not series_data:
        return

    subset_keys = [
        ("All Data", "times_all", "values_all"),
        ("NoFlow", "times_noflow", "values_noflow"),
        ("Leakage", "times_flow", "values_flow"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    colors = plt.cm.tab10.colors

    def _fit_trend(mean_times, mean_values):
        """Compute polynomial trend, return (t_fit, y_fit) or None."""
        if len(mean_times) < 2:
            return None
        t0 = min(mean_times)
        x_num = np.array([(t - t0).total_seconds() for t in mean_times])
        y_arr = np.array(mean_values)
        deg = min(fit_degree, len(mean_times) - 1)
        coeffs = np.polyfit(x_num, y_arr, deg)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(x_num.min(), x_num.max(), 200)
        y_fit = poly(x_fit)
        t_fit = [t0 + pd.Timedelta(seconds=float(s)) for s in x_fit]
        return t_fit, y_fit, deg

    def _collect_means(sd_list, t_key, v_key):
        """Collect per-series means for a subset."""
        mean_times, mean_values = [], []
        for sd in sd_list:
            times = sd.get(t_key, [])
            values = sd.get(v_key, [])
            if not times:
                continue
            t = pd.to_datetime(times, format='ISO8601')
            v = np.asarray(values, dtype=float)
            if len(v) > 0:
                mean_times.append(t.min() + (t.max() - t.min()) / 2)
                mean_values.append(float(np.mean(v)))
        return mean_times, mean_values

    # Pre-compute NoFlow and Leakage trends for overlay on All Data subplot
    nf_means = _collect_means(series_data, "times_noflow", "values_noflow")
    fl_means = _collect_means(series_data, "times_flow", "values_flow")
    nf_trend = _fit_trend(*nf_means)
    fl_trend = _fit_trend(*fl_means)

    for ax, (title, t_key, v_key) in zip(axes, subset_keys):
        ax.set_title(f"{feature_name} — {title}", fontsize=14)
        mean_times = []
        mean_values = []

        for i, sd in enumerate(series_data):
            color = colors[i % len(colors)]
            times = sd.get(t_key, [])
            values = sd.get(v_key, [])
            if not times:
                continue

            t = pd.to_datetime(times, format='ISO8601')
            v = np.asarray(values, dtype=float)

            ax.scatter(t, v, s=4, alpha=0.4, color=color, label=sd['name'])

            if len(v) > 0:
                t_mean = t.min() + (t.max() - t.min()) / 2
                v_mean = float(np.mean(v))
                ax.scatter([t_mean], [v_mean], s=120, color=color,
                           edgecolors='black', linewidths=1.2, zorder=5)
                mean_times.append(t_mean)
                mean_values.append(v_mean)

        trend = _fit_trend(mean_times, mean_values)
        if trend:
            t_fit, y_fit, deg = trend
            ax.plot(t_fit, y_fit, color='orange', linewidth=2.5, zorder=4,
                    label=f'Trend (deg={deg})')

        # Overlay NoFlow + Leakage trends on "All Data" subplot
        if t_key == "times_all":
            if nf_trend:
                ax.plot(nf_trend[0], nf_trend[1], color='dodgerblue',
                        linewidth=2, linestyle='--', zorder=4,
                        label='NoFlow trend')
            if fl_trend:
                ax.plot(fl_trend[0], fl_trend[1], color='red',
                        linewidth=2, linestyle='--', zorder=4,
                        label='Leakage trend')

        ax.set_ylabel(feature_name, fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_segment_feature_drift_overlay(
    feature_name: str,
    series_data_orig: List[dict],
    series_data_tc: List[dict],
    fit_degree: int,
    save_path: Path,
):
    """
    Overlay drift plot: original (solid) vs temperature-corrected (dashed).

    2 subplots:
      1) All Data — per-series scatter with original (solid) and TC (dashed)
         trends, same colour per series.
      2) NoFlow vs Leakage — class-level comparison. NoFlow in blue,
         Leakage in red. Original = solid markers + solid trend,
         TC = diamond markers + dashed trend.

    Args:
        feature_name: Base feature name (without _tc suffix).
        series_data_orig: Series data for the original feature.
        series_data_tc: Series data for the _tc feature.
        fit_degree: Polynomial degree for trend curves.
        save_path: Path to save the figure.
    """
    if not series_data_orig:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    colors = plt.cm.tab10.colors

    def _fit_trend(mean_times, mean_values):
        if len(mean_times) < 2:
            return None
        t0 = min(mean_times)
        x_num = np.array([(t - t0).total_seconds() for t in mean_times])
        y_arr = np.array(mean_values)
        deg = min(fit_degree, len(mean_times) - 1)
        coeffs = np.polyfit(x_num, y_arr, deg)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(x_num.min(), x_num.max(), 200)
        y_fit = poly(x_fit)
        t_fit = [t0 + pd.Timedelta(seconds=float(s)) for s in x_fit]
        return t_fit, y_fit, deg

    def _collect_means(sd_list, t_key, v_key):
        mean_times, mean_values = [], []
        for sd in sd_list:
            times = sd.get(t_key, [])
            values = sd.get(v_key, [])
            if not times:
                continue
            t = pd.to_datetime(times, format='ISO8601')
            v = np.asarray(values, dtype=float)
            if len(v) > 0:
                mean_times.append(t.min() + (t.max() - t.min()) / 2)
                mean_values.append(float(np.mean(v)))
        return mean_times, mean_values

    # Build name→index mapping for consistent colours
    orig_names = [sd['name'] for sd in series_data_orig]
    tc_names = [sd['name'] for sd in series_data_tc] if series_data_tc else []
    all_names = list(dict.fromkeys(orig_names + tc_names))
    name_to_idx = {n: i for i, n in enumerate(all_names)}

    # ── Subplot 1: All Data (per-series colours) ──
    ax1 = axes[0]
    ax1.set_title(f"{feature_name} — All Data", fontsize=14)

    mean_times_o, mean_values_o = [], []
    for sd in series_data_orig:
        ci = name_to_idx[sd['name']]
        color = colors[ci % len(colors)]
        times = sd.get('times_all', [])
        values = sd.get('values_all', [])
        if not times:
            continue
        t = pd.to_datetime(times, format='ISO8601')
        v = np.asarray(values, dtype=float)
        ax1.scatter(t, v, s=4, alpha=0.3, color=color)
        if len(v) > 0:
            t_mean = t.min() + (t.max() - t.min()) / 2
            v_mean = float(np.mean(v))
            ax1.scatter([t_mean], [v_mean], s=120, color=color,
                        edgecolors='black', linewidths=1.2, zorder=5,
                        marker='o', label=sd['name'])
            mean_times_o.append(t_mean)
            mean_values_o.append(v_mean)

    trend_o = _fit_trend(mean_times_o, mean_values_o)
    if trend_o:
        ax1.plot(trend_o[0], trend_o[1], color='orange', linewidth=2.5,
                 zorder=4, linestyle='-', label=f'Original (deg={trend_o[2]})')

    if series_data_tc:
        mean_times_tc, mean_values_tc = [], []
        for sd in series_data_tc:
            ci = name_to_idx[sd['name']]
            color = colors[ci % len(colors)]
            times = sd.get('times_all', [])
            values = sd.get('values_all', [])
            if not times:
                continue
            t = pd.to_datetime(times, format='ISO8601')
            v = np.asarray(values, dtype=float)
            if len(v) > 0:
                t_mean = t.min() + (t.max() - t.min()) / 2
                v_mean = float(np.mean(v))
                ax1.scatter([t_mean], [v_mean], s=100, color=color,
                            edgecolors='black', linewidths=1.2, zorder=5,
                            marker='D')
                mean_times_tc.append(t_mean)
                mean_values_tc.append(v_mean)

        trend_tc = _fit_trend(mean_times_tc, mean_values_tc)
        if trend_tc:
            ax1.plot(trend_tc[0], trend_tc[1], color='green', linewidth=2.5,
                     zorder=4, linestyle='--', label=f'TC (deg={trend_tc[2]})')

    ax1.set_ylabel(feature_name, fontsize=11)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: NoFlow vs Leakage (class colours) ──
    ax2 = axes[1]
    ax2.set_title(f"{feature_name} — NoFlow vs Leakage", fontsize=14)

    CLR_NF = '#1f77b4'   # blue for NoFlow
    CLR_LK = '#d62728'   # red for Leakage

    for data_list, suffix, marker, ls in [
        (series_data_orig, '', 'o', '-'),
        (series_data_tc, ' TC', 'D', '--'),
    ]:
        if not data_list:
            continue
        nf_times, nf_vals = _collect_means(data_list, 'times_noflow', 'values_noflow')
        fl_times, fl_vals = _collect_means(data_list, 'times_flow', 'values_flow')

        # NoFlow markers + trend
        if nf_times:
            ax2.scatter(nf_times, nf_vals, s=100, color=CLR_NF, marker=marker,
                        edgecolors='black', linewidths=1.0, zorder=5,
                        label=f'NoFlow{suffix}')
            trend = _fit_trend(nf_times, nf_vals)
            if trend:
                ax2.plot(trend[0], trend[1], color=CLR_NF, linewidth=2.5,
                         linestyle=ls, zorder=4)

        # Leakage markers + trend
        if fl_times:
            ax2.scatter(fl_times, fl_vals, s=100, color=CLR_LK, marker=marker,
                        edgecolors='black', linewidths=1.0, zorder=5,
                        label=f'Leakage{suffix}')
            trend = _fit_trend(fl_times, fl_vals)
            if trend:
                ax2.plot(trend[0], trend[1], color=CLR_LK, linewidth=2.5,
                         linestyle=ls, zorder=4)

    ax2.set_ylabel(feature_name, fontsize=11)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_feature_drift_overview(
    feature_name: str,
    series_data: List[dict],
    save_path: Path,
):
    """
    Plot per-series flow/noflow means over time for a single feature.

    Shows absolute drift and class separability across measurement series.

    Args:
        feature_name: Name of the feature being plotted.
        series_data: List of dicts with keys 'name', 'mid_time' (datetime-like),
            'flow_mean', 'flow_std', 'noflow_mean', 'noflow_std'.
        save_path: Path to save the figure.
    """
    if not series_data:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Feature Drift: {feature_name}", fontsize=14)

    mid_times = [pd.to_datetime(sd['mid_time']) for sd in series_data]
    flow_means = np.array([sd['flow_mean'] for sd in series_data])
    flow_stds = np.array([sd['flow_std'] for sd in series_data])
    noflow_means = np.array([sd['noflow_mean'] for sd in series_data])
    noflow_stds = np.array([sd['noflow_std'] for sd in series_data])
    names = [sd['name'] for sd in series_data]

    # Plot noflow markers with error bars
    ax.errorbar(mid_times, noflow_means, yerr=noflow_stds,
                fmt='s', color='#6495ED', markersize=8, capsize=5,
                linewidth=1.2, label='noflow', zorder=5)

    # Plot flow markers with error bars
    ax.errorbar(mid_times, flow_means, yerr=flow_stds,
                fmt='o', color='#E8A0BF', markersize=8, capsize=5,
                linewidth=1.2, label='flow', zorder=5)

    # Trend lines (linear fit) if enough points
    if len(mid_times) >= 2:
        t0 = min(mid_times)
        x_num = np.array([(t - t0).total_seconds() for t in mid_times])

        for y_vals, color, ls, lbl in [
            (flow_means, '#E8A0BF', '--', 'flow trend'),
            (noflow_means, '#6495ED', '--', 'noflow trend'),
        ]:
            mask = np.isfinite(y_vals)
            if mask.sum() >= 2:
                coeffs = np.polyfit(x_num[mask], y_vals[mask], 1)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(x_num.min(), x_num.max(), 200)
                y_fit = poly(x_fit)
                t_fit = [t0 + pd.Timedelta(seconds=float(s)) for s in x_fit]
                ax.plot(t_fit, y_fit, color=color, linewidth=1.5,
                        linestyle=ls, alpha=0.7, label=lbl, zorder=4)

    # Annotate series names
    for t, name in zip(mid_times, names):
        ax.axvline(t, color='gray', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(feature_name, fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_feature_drift_summary(
    feature_stats: Dict[str, List[dict]],
    save_path: Path,
):
    """
    Plot a heatmap summary of feature drift across series.

    Shows z-scored feature means per series for flow and noflow classes
    side by side.

    Args:
        feature_stats: Mapping of feature_name -> list of series dicts
            (same format as plot_feature_drift_overview series_data).
        save_path: Path to save the figure.
    """
    if not feature_stats:
        return

    feature_names = sorted(feature_stats.keys())
    # Use series order from the first feature (all should be the same)
    first_data = feature_stats[feature_names[0]]
    series_names = [sd['name'] for sd in first_data]

    n_features = len(feature_names)
    n_series = len(series_names)

    if n_features == 0 or n_series == 0:
        return

    # Build matrices: rows = features, cols = series
    flow_matrix = np.full((n_features, n_series), np.nan)
    noflow_matrix = np.full((n_features, n_series), np.nan)

    series_idx = {name: i for i, name in enumerate(series_names)}

    for fi, feat in enumerate(feature_names):
        for sd in feature_stats.get(feat, []):
            si = series_idx.get(sd['name'])
            if si is not None:
                flow_matrix[fi, si] = sd['flow_mean']
                noflow_matrix[fi, si] = sd['noflow_mean']

    # Z-score each feature (row) across series for comparability
    def zscore_rows(mat):
        out = np.full_like(mat, np.nan)
        for i in range(mat.shape[0]):
            row = mat[i]
            valid = np.isfinite(row)
            if valid.sum() >= 2:
                mu = np.nanmean(row)
                sigma = np.nanstd(row)
                if sigma > 0:
                    out[i] = (row - mu) / sigma
                else:
                    out[i] = 0.0
            elif valid.sum() == 1:
                out[i] = 0.0
        return out

    flow_z = zscore_rows(flow_matrix)
    noflow_z = zscore_rows(noflow_matrix)

    # Determine shared color range
    vmax = max(np.nanmax(np.abs(flow_z)) if np.any(np.isfinite(flow_z)) else 1.0,
               np.nanmax(np.abs(noflow_z)) if np.any(np.isfinite(noflow_z)) else 1.0)
    vmax = max(vmax, 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, 3 * n_series), max(6, 0.4 * n_features)))
    fig.suptitle("Feature Drift Summary", fontsize=14)

    for ax, mat, title in [(ax1, noflow_z, 'noflow'), (ax2, flow_z, 'flow')]:
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(n_series))
        ax.set_xticklabels(series_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names, fontsize=7)

    fig.colorbar(im, ax=[ax1, ax2], label='z-score', shrink=0.8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# =============================================================================
# DEPRECATED PLOTS
# =============================================================================

def plot_blocks_overview(
    df_blocks: pd.DataFrame,
    jump_block_indices: np.ndarray,
    title: str,
    save_path: Optional[Path] = None
):
    """
    Plot block means with jump markers and relaxation coloring.

    DEPRECATED: Use plot_relaxations_overview() instead.

    Args:
        df_blocks: Block-level DataFrame with relaxation_id column.
        jump_block_indices: Array of block indices where jumps occur.
        title: Plot title.
        save_path: Optional path to save the plot.
    """
    if len(df_blocks) == 0:
                return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{title} - Block-Level Analysis", fontsize=14)

    ax1 = axes[0]

    colors = plt.cm.tab10.colors

    for rel_id in df_blocks['relaxation_id'].unique():
        if rel_id == -1:
            continue

        rel_mask = df_blocks['relaxation_id'] == rel_id
        rel_blocks = df_blocks[rel_mask]
        color = colors[rel_id % len(colors)]

        label = None
        if rel_id < 10:
            label = f'Relaxation {rel_id}'

        ax1.plot(rel_blocks['block_time'], rel_blocks['block_mean'],
                marker='o', markersize=3, linewidth=1, color=color,
                linestyle='-', label=label)

    unassigned = df_blocks[df_blocks['relaxation_id'] == -1]
    if len(unassigned) > 0:
        ax1.plot(unassigned['block_time'], unassigned['block_mean'],
                'o', markersize=2, color='gray', alpha=0.5)

    for jump_idx in jump_block_indices:
        if jump_idx < len(df_blocks):
            jump_time = df_blocks['block_time'].iloc[jump_idx]
            ax1.axvline(x=jump_time, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('Block Mean', fontsize=11)
    ax1.grid(True, alpha=0.3)

    n_relaxations = df_blocks['relaxation_id'].nunique() - (1 if -1 in df_blocks['relaxation_id'].values else 0)
    ax1.text(0.02, 0.98,
            f'Blocks: {len(df_blocks)}\n'
            f'Jumps: {len(jump_block_indices)}\n'
            f'Relaxations: {n_relaxations}',
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]

    if len(df_blocks) > 1:
        diffs = np.diff(df_blocks['block_mean'].values)
        diff_times = df_blocks['block_time'].iloc[1:].values

        ax2.plot(diff_times, diffs, 'b-', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff))
        robust_std = 1.4826 * mad
        threshold = 5.0 * robust_std

        ax2.axhline(y=median_diff + threshold, color='red', linestyle=':',
                   linewidth=1, alpha=0.7)
        ax2.axhline(y=median_diff - threshold, color='red', linestyle=':',
                   linewidth=1, alpha=0.7)
        ax2.axhline(y=median_diff, color='green', linestyle='--',
                   linewidth=1, alpha=0.7)

        for jump_idx in jump_block_indices:
            if jump_idx > 0 and jump_idx <= len(diffs):
                ax2.axvline(x=diff_times[jump_idx-1], color='red',
                           linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Block-to-Block Diff', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close('all')


# =============================================================================
# FEATURE CORRELATION PLOTS
# =============================================================================

def plot_feature_group_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_name: str,
    series_name: str,
    save_path: Path,
) -> None:
    """
    Plot correlation heatmap for a group of features within one series.

    Args:
        df: DataFrame containing the feature columns.
        feature_cols: List of column names for this feature group.
        group_name: Human-readable group name (used in title).
        series_name: Name of the series (used in title).
        save_path: Path to save the plot.
    """
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return

    corr = df[available].corr()

    # Short labels: strip common prefixes for readability
    short_labels = []
    for c in available:
        for prefix in ('interval_up_', 'interval_down_', 'runtime_diff_',
                       'norm_interval_up_', 'norm_interval_down_',
                       'norm_interval_diff_', 't_up_', 't_down_'):
            if c.startswith(prefix):
                c = c[len(prefix):]
                break
        # Remove unit suffix
        c = c.replace(' in ps', '').replace(' in mV', '')
        short_labels.append(c)

    n = len(available)
    fig_size = max(6, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=max(6, 10 - n // 4), color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    ax.set_title(f'{group_name} — {series_name}', fontsize=13)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close('all')
