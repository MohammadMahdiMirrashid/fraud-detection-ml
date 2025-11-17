# Architecture Overview

## System Design

This fraud detection system follows a modular, production-ready architecture:

### Data Pipeline
1. **Data Generation** (`src/data/simulate_data.py`)
   - Synthetic transaction generation
   - Fraud pattern injection
   - Data validation

2. **Preprocessing** (`src/data/preprocess.py`)
   - Data cleaning
   - Missing value handling
   - Outlier treatment

3. **Feature Engineering** (`src/data/feature_engineering.py`)
   - Time-based features
   - Rolling window aggregations
   - Customer-level aggregates
   - Velocity features
   - Risk heuristics

### Modeling Pipeline
1. **Model Training** (`src/models/trainer.py`)
   - Unified API for all models
   - Cross-validation support
   - Model persistence

2. **Model Types**
   - Isolation Forest (unsupervised)
   - Autoencoder (deep learning)
   - Gradient Boosting (LightGBM/XGBoost/CatBoost)

### Evaluation & Optimization
1. **Metrics** (`src/evaluation/metrics.py`)
   - PR-AUC, ROC-AUC
   - Precision@K, Recall@K
   - Cost-based metrics

2. **Thresholding** (`src/models/thresholding.py`)
   - Cost-based optimization
   - Business constraint handling

3. **Calibration** (`src/evaluation/calibration.py`)
   - Platt scaling
   - Isotonic regression
   - Brier score, ECE

### Explainability
1. **SHAP Integration** (`src/explainability/shap_utils.py`)
   - Global feature importance
   - Local explanations
   - Visualization tools

2. **Audit Reports** (`src/explainability/report_generator.py`)
   - Transaction-level explanations
   - Batch reporting

### Real-Time Scoring
1. **Streaming** (`src/streaming/`)
   - Mock event stream
   - Kafka integration (optional)
   - Real-time feature generation
   - Online scoring

## Workflow

```
Raw Data → Preprocessing → Feature Engineering → Model Training
                                                         ↓
Real-Time Events → Feature Extraction → Scoring → Alert Generation
                                                         ↓
                                              Explainability → Audit Reports
```

