

# Ultrasonic Micro-Leak Detection Using Machine Learning

Machine learning and deep learning pipeline for detecting micro-leaks in industrial pipe systems using ultrasonic clamp-on sensor measurements.


## Technologies

- **Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Deep Learning:** PyTorch, Transformer Neural Networks
- **Data Processing:** Pandas, NumPy, Signal Processing
- **Visualization:** Matplotlib

## Key Features

- Preprocessing and normalization of multivariate ultrasonic sensor data
- Automated equilibrium detection and time-series feature extraction
- Time-series patch construction for deep learning
- XGBoost and Transformer-based leak classification
- Chronological train/test separation to evaluate generalization
- Model evaluation using accuracy, precision, recall, and F1-score

## Results

| Model | Test Accuracy | Precision | Recall |
|---|---|---|---|
| XGBoost | 88.26% | — | 100% |
| Transformer | 82.08% | 100% | 72.69% |

## Project Structure

```text
├── configs/        # Configuration files
├── scripts/        # Training and evaluation scripts
├── src/            # Source code
├── setup.py        # Package configuration
└── results.md      # Experimental results
```

## Author

**Niloufar Mashhadi Gholamali**

M.Sc. Computer & Systems Engineering (AI) — Technische Universität Ilmenau
