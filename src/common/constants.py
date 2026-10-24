"""
Constants for KILD preprocessing pipeline.

Contains column definitions and feature names used across all preprocessing steps.
"""

# Feature columns
UP_COLS = [f"t_up_{i} in ps" for i in range(10)]
DOWN_COLS = [f"t_down_{i} in ps" for i in range(10)]
UP_COLS_NORM = [f"t_up_{i} norm" for i in range(10)]
DOWN_COLS_NORM = [f"t_down_{i} norm" for i in range(10)]

FEATURES = (
    UP_COLS + DOWN_COLS
    + ['pw_up', 'pw_down', 'amplitude_up in mV', 'amplitude_down in mV', 'tdiff in ps']
)

TEMP_COLS = ['Temp Pipe in °C', 'Temp Ambient in °C']
NUMERICAL_COLS = FEATURES + TEMP_COLS

# Columns added by the pipeline steps
PIPELINE_COLS = [
    'sample',             # label step: series name
    'flow_type',          # label step: flow / preleakage / postleakage / noflow / undefined
    'valid',              # clean step: 1 (inlier), 0 (local outlier), -1 (global outlier)
    'relaxation_id',      # extract step: int >= 0 or -1
    'valid_relaxation',   # extract step: 0 / 1
    'equilibrium',        # equilibrium step: 0 / 1
]
