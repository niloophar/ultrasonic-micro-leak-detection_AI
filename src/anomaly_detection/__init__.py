"""
Anomaly detection module for unsupervised leakage detection.

Provides a pipeline for:
1. Training CNN-based AutoEncoder or VAE on NoFlow (normal) data
2. Detecting leakage anomalies via reconstruction error, latent distance, or combined score
3. Latent space visualization for interpretability
"""
