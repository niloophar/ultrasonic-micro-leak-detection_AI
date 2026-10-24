"""
Preprocessing pipeline for KILD project.

This package contains the preprocessing steps:
1. label: Label data as flow/noflow based on leakage rules
2. clean: Detect and mark outliers (global and local)
3. relaxation: Extract thermal relaxation segments
4. equilibrium: Detect thermal equilibrium regions

Usage:
    python -m src.preprocessing.main --run label clean relaxation equilibrium
"""

from .label import process_label
from .clean import process_clean
from .relaxation import process_relaxation
from .equilibrium import process_equilibrium

__all__ = [
    "process_label",
    "process_clean",
    "process_relaxation",
    "process_equilibrium",
]
