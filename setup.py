"""
Setup script for fraud-detection-ml package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="fraud-detection-ml",
    version="0.1.0",
    description="ML-Driven Fraud Detection on Synthetic Banking Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/fraud-detection-ml",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "ml": [
            "lightgbm>=3.3.0",
            "xgboost>=1.5.0",
            "catboost>=1.0.0",
        ],
        "deep": [
            "tensorflow>=2.8.0",
            "keras>=2.8.0",
        ],
        "explain": [
            "shap>=0.40.0",
        ],
        "streaming": [
            "kafka-python>=2.0.0",
        ],
        "data": [
            "sdv>=0.15.0",
            "ctgan>=0.7.0",
            "tsfresh>=0.19.0",
            "pytz>=2021.3",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

