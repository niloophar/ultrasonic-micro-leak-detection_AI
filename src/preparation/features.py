"""
Feature Engineering Module - Step 2 of Classification Pipeline (Milestone C2).

Computes temperature-invariant engineered features from raw ultrasound
transit time measurements:

Group 1 - Interval Features (Within-Stream Deltas, cancel offset b):
- 9 interval_up (consecutive zero-crossing differences, upstream)
- 9 interval_down (consecutive zero-crossing differences, downstream)
- 1 period_sum_up (sum of upstream intervals = t_up_9 - t_up_0)
- 1 period_sum_down (sum of downstream intervals = t_down_9 - t_down_0)
- 1 period_asymmetry (|period_sum_up - period_sum_down|)

Group 2 - Runtime Difference Features (Cross-Stream, cancel offset b):
- 10 runtime_diff (|t_up_i - t_down_i| per zero-crossing)
- 1 total_runtime_diff (sum of all runtime_diff_i)

Group 3 - Combined Temperature-Invariant Feature (cancels a and b):
- 1 normalized_period_asymmetry (period_asymmetry / total_runtime_diff)

Group 4 - Sensor Signal Difference Features:
- 1 amplitude_diff (|amplitude_up - amplitude_down|)
- 1 pw_diff (|pw_up - pw_down|)

Group 5 - Normalized Interval Features (cancel both a and b):
- 9 norm_interval_up (interval_up_i / period_sum_up, dimensionless fractions)
- 9 norm_interval_down (interval_down_i / period_sum_down)

Group 6 - Normalized Interval Difference Features (cancel both a and b):
- 9 norm_interval_diff (norm_interval_up_i - norm_interval_down_i)
  Removes the geometric background D_i/S that dominates norm_interval values,
  isolating the per-echo flow signature. Zero without flow, non-zero with flow.

Group 7 - Echo Ratio Features (cancel a_i(T) exactly):
- 9 echo_ratio (interval_up_i / interval_down_i, dimensionless)
  Up and down traverse the same medium in opposite directions, so the
  per-echo temperature factor a_i(T) cancels exactly. Ratio ≈ 1.0 without
  flow, deviates from 1.0 with flow.
- 1 period_ratio (period_sum_up / period_sum_down, aggregated echo ratio)

Group 9 - Total-Length Combinations (operate on full traversal time):
  Micro-leaks and flow at low rate barely perturb per-ZC intervals — the
  per-echo flow signature is buried in noise and temperature drift. The
  TOTAL transit times (= telescoping sum of the intervals, equivalent to
  t_9 - t_0) carry a much larger signal and combine Up+Down directly.
- 1 period_sum_total (period_sum_up + period_sum_down, mean traversal proxy)
- 1 period_diff_signed (period_sum_up - period_sum_down, signed; keeps
  the directional information that period_asymmetry discards via abs)
- 1 flow_index ((up - down) / (up + down), dimensionless; the classical
  transit-time differential flow ratio — amplitude-independent, drift-robust)
- 1 period_sum_total_per_tpipe (sum / Temp Pipe, temperature-normalized
  traversal length, suppresses the dominant thermal drift)
- 1 period_diff_per_tpipe (signed diff / Temp Pipe)
- 1 flow_index_per_tpipe (flow_index / Temp Pipe)

Total: 79 features

Temperature robustness:
- Group 1+2 features cancel additive offset b
- Group 3+5+6 features also cancel multiplicative factor a (global)
- Group 7 features cancel per-echo multiplicative factor a_i(T) exactly
- Group 9 flow_index is dimensionless and doubly normalised; the
  _per_tpipe variants divide out the residual linear thermal trend
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.constants import UP_COLS, DOWN_COLS


# Feature column name definitions
INTERVAL_UP_COLS = [f"interval_up_{i}" for i in range(9)]
INTERVAL_DOWN_COLS = [f"interval_down_{i}" for i in range(9)]
NORM_INTERVAL_UP_COLS = [f"norm_interval_up_{i}" for i in range(9)]
NORM_INTERVAL_DOWN_COLS = [f"norm_interval_down_{i}" for i in range(9)]
NORM_INTERVAL_DIFF_COLS = [f"norm_interval_diff_{i}" for i in range(9)]
RUNTIME_DIFF_COLS = [f"runtime_diff_{i}" for i in range(10)]
ECHO_RATIO_COLS = [f"echo_ratio_{i}" for i in range(9)]

# Group 9 — total-length combinations (see module docstring).
TOTAL_LENGTH_COLS = [
    'period_sum_total',
    'period_diff_signed',
    'flow_index',
    'period_sum_total_per_tpipe',
    'period_diff_per_tpipe',
    'flow_index_per_tpipe',
]

ALL_FEATURE_COLS = (
    INTERVAL_UP_COLS
    + INTERVAL_DOWN_COLS
    + NORM_INTERVAL_UP_COLS
    + NORM_INTERVAL_DOWN_COLS
    + NORM_INTERVAL_DIFF_COLS
    + ['period_sum_up', 'period_sum_down']
    + ['period_asymmetry']
    + RUNTIME_DIFF_COLS
    + ['total_runtime_diff']
    + ['normalized_period_asymmetry']
    + ECHO_RATIO_COLS
    + ['period_ratio']
    + TOTAL_LENGTH_COLS
    + ['temp_diff']
    + ['amplitude_diff', 'pw_diff']
)

N_FEATURES = len(ALL_FEATURE_COLS)  # 79


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 72 engineered features for each data point.

    The features are designed to cancel out temperature-dependent offsets:
    - Differences cancel additive offset b
    - Normalized features and ratios also cancel multiplicative factor a
    - Echo ratios cancel per-echo a_i(T) exactly

    Args:
        df: DataFrame containing the raw transit time columns
            (t_up_0..t_up_9, t_down_0..t_down_9 in ps)
            and optionally amplitude_up/down, pw_up/down.

    Returns:
        DataFrame with 72 feature columns (same number of rows as input).
    """
    result = pd.DataFrame(index=df.index)

    # 1. Interval features: consecutive zero-crossing differences (9 each)
    for i in range(9):
        result[INTERVAL_UP_COLS[i]] = (
            df[UP_COLS[i + 1]].values - df[UP_COLS[i]].values
        )
        result[INTERVAL_DOWN_COLS[i]] = (
            df[DOWN_COLS[i + 1]].values - df[DOWN_COLS[i]].values
        )

    # 2. Summed half-period intervals (= t_9 - t_0 by telescoping)
    result['period_sum_up'] = result[INTERVAL_UP_COLS].sum(axis=1)
    result['period_sum_down'] = result[INTERVAL_DOWN_COLS].sum(axis=1)

    # 2b. Normalized interval features: interval_i / period_sum
    #     Cancels both a(T) and b(T), giving dimensionless fractions.
    psu = result['period_sum_up'].values.copy()
    psu[psu == 0] = 1.0
    psd = result['period_sum_down'].values.copy()
    psd[psd == 0] = 1.0
    for i in range(9):
        result[NORM_INTERVAL_UP_COLS[i]] = (
            result[INTERVAL_UP_COLS[i]].values / psu
        )
        result[NORM_INTERVAL_DOWN_COLS[i]] = (
            result[INTERVAL_DOWN_COLS[i]].values / psd
        )

    # 2c. Normalized interval differences: norm_up_i - norm_down_i
    #     Removes geometric background D_i/S, isolates per-echo flow signature.
    #     Zero without flow, non-zero with flow. Cancels both a(T) and b(T).
    for i in range(9):
        result[NORM_INTERVAL_DIFF_COLS[i]] = (
            result[NORM_INTERVAL_UP_COLS[i]].values
            - result[NORM_INTERVAL_DOWN_COLS[i]].values
        )

    # 3. Period asymmetry: |period_sum_up - period_sum_down|
    result['period_asymmetry'] = np.abs(
        result['period_sum_up'].values - result['period_sum_down'].values
    )

    # 4. Runtime differences: |t_up_i - t_down_i| per zero-crossing (10)
    for i in range(10):
        result[RUNTIME_DIFF_COLS[i]] = np.abs(
            df[UP_COLS[i]].values - df[DOWN_COLS[i]].values
        )

    # 5. Total runtime difference: sum of all |t_up_i - t_down_i|
    result['total_runtime_diff'] = result[RUNTIME_DIFF_COLS].sum(axis=1)

    # 6. Normalized period asymmetry (fully temperature-invariant)
    # = period_asymmetry / total_runtime_diff
    denom = result['total_runtime_diff'].values.copy()
    denom[denom == 0] = 1.0
    result['normalized_period_asymmetry'] = (
        result['period_asymmetry'].values / denom
    )
    result.loc[result['total_runtime_diff'].values == 0, 'normalized_period_asymmetry'] = 0.0

    # 7. Echo ratio features: interval_up_i / interval_down_i
    #    Temperature-invariant because up and down traverse the same medium
    #    in opposite directions, so a_i(T) cancels exactly.
    #    Without flow: ratio ≈ 1.0 (symmetric paths)
    #    With flow: ratio ≠ 1.0 (Doppler breaks symmetry)
    for i in range(9):
        num = result[INTERVAL_UP_COLS[i]].values.astype(float)
        den = result[INTERVAL_DOWN_COLS[i]].values.copy().astype(float)
        den[den == 0] = 1.0
        result[ECHO_RATIO_COLS[i]] = num / den

    # 7b. Period ratio: period_sum_up / period_sum_down (aggregated echo ratio)
    psu_ratio = result['period_sum_up'].values.astype(float)
    psd_ratio = result['period_sum_down'].values.copy().astype(float)
    psd_ratio[psd_ratio == 0] = 1.0
    result['period_ratio'] = psu_ratio / psd_ratio

    # 8. Group 9 — Total-length combinations.
    # Operate on period_sum_{up,down}, which by telescoping equal t_9 - t_0 and
    # so carry the full traversal-length signal instead of the noisy per-ZC one.
    psu_tot = result['period_sum_up'].values.astype(float)
    psd_tot = result['period_sum_down'].values.astype(float)
    period_sum_total = psu_tot + psd_tot
    period_diff_signed = psu_tot - psd_tot
    result['period_sum_total'] = period_sum_total
    result['period_diff_signed'] = period_diff_signed

    # Dimensionless flow ratio: (up - down) / (up + down).
    # Classical transit-time differential measurement — amplitude-agnostic.
    sum_safe = period_sum_total.copy()
    sum_safe[sum_safe == 0] = 1.0
    result['flow_index'] = period_diff_signed / sum_safe

    # Temperature-normalised variants. Divide by absolute pipe temperature —
    # the 8-series drift study showed temp_pipe explains the centroid drift on
    # PC2 with ρ=-0.90, so normalising by it removes the dominant linear trend.
    if 'Temp Pipe in \u00b0C' in df.columns:
        t_pipe = df['Temp Pipe in \u00b0C'].values.astype(float)
        t_pipe_safe = t_pipe.copy()
        t_pipe_safe[t_pipe_safe == 0] = 1.0
        result['period_sum_total_per_tpipe'] = period_sum_total / t_pipe_safe
        result['period_diff_per_tpipe'] = period_diff_signed / t_pipe_safe
        result['flow_index_per_tpipe'] = result['flow_index'].values / t_pipe_safe

    # 9. Temperature difference feature
    if 'Temp Pipe in \u00b0C' in df.columns and 'Temp Ambient in \u00b0C' in df.columns:
        result['temp_diff'] = (
            df['Temp Pipe in \u00b0C'].values - df['Temp Ambient in \u00b0C'].values
        )

    # 9. Sensor signal difference features
    if 'amplitude_up in mV' in df.columns and 'amplitude_down in mV' in df.columns:
        result['amplitude_diff'] = np.abs(
            df['amplitude_up in mV'].values - df['amplitude_down in mV'].values
        )
    if 'pw_up' in df.columns and 'pw_down' in df.columns:
        result['pw_diff'] = np.abs(
            df['pw_up'].values - df['pw_down'].values
        )

    return result


def fit_temperature_correction(
    feature_values: Dict[str, np.ndarray],
    temperatures: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Fit linear temperature correction coefficients from NoFlow data.

    For each feature, fits: feature = alpha * T + beta (on NoFlow data).
    Correction applied later: feature_tc = feature - alpha * (T - T_ref)
    where T_ref = mean(T_noflow). This removes the linear temperature
    trend while preserving the mean feature level.

    Args:
        feature_values: Dict mapping feature name to array of NoFlow values.
        temperatures: Array of NoFlow temperature values (same length).

    Returns:
        Dict mapping feature name to {'alpha', 'beta', 'T_ref'}.
    """
    T_ref = float(np.mean(temperatures))
    T_var = float(np.var(temperatures))
    coefficients: Dict[str, Dict[str, float]] = {}

    for feat_name, values in feature_values.items():
        if len(values) < 2 or T_var == 0:
            continue
        alpha = float(np.cov(values, temperatures)[0, 1] / T_var)
        beta = float(np.mean(values) - alpha * np.mean(temperatures))
        coefficients[feat_name] = {
            'alpha': alpha,
            'beta': beta,
            'T_ref': T_ref,
        }

    return coefficients


def apply_temperature_correction(
    features_df: pd.DataFrame,
    temperatures: np.ndarray,
    coefficients: Dict[str, Dict[str, float]],
    suffix: str = '_tc',
) -> pd.DataFrame:
    """Apply linear temperature correction to feature columns.

    For each feature with fitted coefficients:
        feature_tc = feature - alpha * (T - T_ref)

    Args:
        features_df: DataFrame with feature columns.
        temperatures: Temperature array (same length as features_df).
        coefficients: Output from fit_temperature_correction.
        suffix: Suffix for corrected column names (default '_tc').

    Returns:
        DataFrame with additional corrected columns appended.
    """
    result = features_df.copy()
    for feat_name, coeff in coefficients.items():
        if feat_name in result.columns:
            alpha = coeff['alpha']
            T_ref = coeff['T_ref']
            result[f"{feat_name}{suffix}"] = (
                result[feat_name].values - alpha * (temperatures - T_ref)
            )
    return result


def get_feature_columns(selection: str = 'all') -> List[str]:
    """
    Get the list of feature column names based on selection.

    Args:
        selection: 'all' for all 35 features, or 'compact' for a reduced set
            of 21 features (9 interval_up + 9 interval_down
            + period_asymmetry + total_runtime_diff
            + normalized_period_asymmetry).

    Returns:
        List of feature column names.
    """
    if selection == 'compact':
        return (
            INTERVAL_UP_COLS
            + INTERVAL_DOWN_COLS
            + ['period_asymmetry', 'total_runtime_diff', 'normalized_period_asymmetry']
        )
    return list(ALL_FEATURE_COLS)
