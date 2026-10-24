"""
Preparation module for steady-state leakage detection data (Milestone C).

Provides a pipeline for:
1. Extracting equilibrium data for flow/noflow classes
2. Computing engineered features (42 temperature-invariant features)
3. Preparing data: downsample, cut patches, center (saved as .npy per segment)
4. Splitting segments into train/val/test splits
"""
