"""Setup script for leak detection project."""
from setuptools import setup, find_packages

setup(
    name="leak-detection",
    version="0.1.0",
    description="Robust micro-leak detection in real-world pipe systems",
    author="Niloufar Mashhadi Gholamali",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
)
