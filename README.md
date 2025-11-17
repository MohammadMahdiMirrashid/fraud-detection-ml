# ML-Driven Fraud Detection on Synthetic Banking Data

Fraud detection is a high-impact machine learning problem where data imbalance, anomaly modeling, threshold selection, and explainability matter far more than raw accuracy.  
This project demonstrates an end-to-end, production-minded fraud detection system using **synthetic banking transactions**, showcasing both **ML science** and **engineering** competencies.

---

## 📊 Key ML Challenges Addressed

### **✔ Class imbalance (0.5% fraud rate)**  
Shows techniques like:
- stratified sampling  
- anomaly detection baselines  
- evaluation with PR-AUC, Precision@K  

### **✔ High-stakes thresholding**  
Accuracy is useless here.  
We optimize:
- False positive cost  
- Fraud loss cost  
- Minimal investigator workload  

### **✔ Explainability & auditability**  
Crucial in real fin-tech systems.

---

## 🧪 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Generate synthetic data
```bash
python src/data/simulate_data.py
```

### Train models (example)
```bash
python src/models/trainer.py --model gradient_boosting
```

### Run streaming demo (optional)
```bash
python src/streaming/mock_stream.py
```

---

## 📈 Example Outputs Included
- PR curves  
- ROC curves  
- Calibration plots  
- SHAP summaries  
- Fraud risk dashboard (optional)  

---

## 📄 Why This Project Matters
This project captures the *real problems* ML scientists face in fraud detection:
- Noisy, imbalanced data  
- Need for feature creativity  
- Need for calibrated models  
- False positives are expensive  
- Explanations required by financial regulators  
- Real-time scoring constraints  

Exactly the kind of complexity hiring managers want to see.

---

## 🚀 Project Highlights

### **1. Synthetic Data Generation**
- CTGAN / Gaussian Copulas for realistic distributions  
- Rule-based pattern injection for fraudulent behavior  
- Configurable fraud rate (default: 0.5%)  
- Reproducible pipeline saving raw → cleaned → ML-ready data  

### **2. Heavy Feature Engineering**
Includes both customer-level and transaction-level features:
- Rolling windows (1h, 6h, 24h)  
- Statistical aggregates  
- Velocity, frequency, and burst features  
- Risk scoring heuristics  
- Outlier-based synthetic indicators

### **3. Multiple Modeling Strategies**
A comparison between:
- **Isolation Forest** (anomaly detection)  
- **Autoencoders** (deep unsupervised)  
- **Gradient Boosting Models** (LightGBM/XGBoost/CatBoost)  

Unified training API (`src/models/trainer.py`).

### **4. Threshold Optimization & Calibration**
Fraud detection success is about setting the right alert threshold:
- ROC/PR curve analysis  
- Precision-at-K  
- Cost-based optimization (fraud loss vs. investigation cost)  
- Platt scaling & isotonic regression for calibrated probabilities

### **5. Explainability for Audit & Risk Teams**
- SHAP values for boosting models  
- Feature attribution visualizations  
- Example audit-style report  
- Model transparency considerations for regulators

### **6. Optional Real-Time Streaming Demo**
- Mock event loop OR Kafka producer/consumer  
- Sliding window feature generation in real time  
- Online scoring pipeline (load model → classify events)  

---

## 🧱 Project Structure

```
fraud-detection-ml/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_thresholding_and_calibration.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_streaming_demo.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── simulate_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py
│   │   ├── autoencoder.py
│   │   ├── gradient_boosting.py
│   │   ├── trainer.py
│   │   └── thresholding.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── calibration.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_utils.py
│   │   ├── feature_importance.py
│   │   └── report_generator.py
│   │
│   └── streaming/
│       ├── __init__.py
│       ├── mock_stream.py
│       ├── kafka_producer.py
│       ├── kafka_consumer.py
│       └── real_time_scoring.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_simulation.py
│   ├── test_feature_eng.py
│   ├── test_models.py
│   └── test_thresholding.py
│
└── docs/
    ├── architecture.md
    ├── feature_dict.md
    ├── explainability_report_example.md
    └── streaming_design.md
```

---

## 📜 License
MIT License.

---

## 🤝 Contributions
Issues and pull requests are welcome.

