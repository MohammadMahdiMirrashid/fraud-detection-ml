# Fraud Detection ML

ML-Driven Fraud Detection on Synthetic Banking Data

A comprehensive machine learning framework for detecting fraudulent transactions in banking data. This project provides tools for data simulation, model training, real-time scoring, and model explainability.

## Features

- **Synthetic Data Generation**: Create realistic banking transaction data with configurable fraud patterns
- **Multiple ML Models**: Support for LightGBM, XGBoost, CatBoost, and deep learning models
- **Real-Time Streaming**: Process and score transactions in real-time using Kafka or mock event streams
- **Model Explainability**: SHAP-based explanations for model predictions, suitable for audit and regulatory compliance
- **Comprehensive Testing**: Unit tests for data simulation and model components
- **Jupyter Notebooks**: Interactive demos and tutorials

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# Install with ML models (LightGBM, XGBoost, CatBoost)
pip install -e ".[ml]"

# Install with deep learning support
pip install -e ".[deep]"

# Install with explainability tools (SHAP)
pip install -e ".[explain]"

# Install with streaming support (Kafka)
pip install -e ".[streaming]"

# Install with data generation tools
pip install -e ".[data]"

# Install with development tools
pip install -e ".[dev]"

# Install all extras
pip install -e ".[ml,deep,explain,streaming,data,dev]"
```

## Requirements

- Python >= 3.8
- See `setup.py` for complete dependency list

## Project Structure

```
fraud-detection-ml/
├── src/
│   ├── data/
│   │   └── simulate_data.py      # Transaction data simulation
│   ├── streaming/
│   │   ├── mock_stream.py         # Mock event stream
│   │   └── real_time_scoring.py   # Real-time scoring engine
│   └── explainability/
│       ├── shap_utils.py          # SHAP explanation utilities
│       └── report_generator.py    # Audit report generation
├── notebooks/
│   ├── 05_explainability.ipynb    # Model explainability demo
│   └── 06_streaming_demo.ipynb    # Real-time streaming demo
├── tests/
│   └── test_data_simulation.py    # Unit tests
├── configs/
│   └── defaults.yaml              # Configuration files
├── setup.py                       # Package setup
└── README.md                      # This file
```

## Quick Start

### 1. Generate Synthetic Data

```python
from src.data.simulate_data import TransactionSimulator

# Create a transaction simulator
simulator = TransactionSimulator(
    n_customers=1000,
    n_transactions=10000,
    fraud_rate=0.01,
    random_state=42
)

# Generate synthetic transaction data
df = simulator.generate()
```

### 2. Train a Model

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import joblib

# Prepare features and target
X = df.drop(['fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1)
y = df['fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/best_model.pkl')
```

### 3. Real-Time Scoring

```python
from src.streaming.real_time_scoring import RealTimeScorer
from src.streaming.mock_stream import EventStream

# Initialize scorer
scorer = RealTimeScorer(
    model_path='models/best_model.pkl',
    model_type='lightgbm',
    threshold=0.5
)

# Create event stream
stream = EventStream(rate_per_second=5, n_events=100, fraud_rate=0.01)

# Process events in real-time
scorer.process_stream(stream, max_events=100)
```

### 4. Model Explainability

```python
from src.explainability.shap_utils import explain_model_shap, plot_shap_summary

# Generate SHAP explanations
explainer, shap_values = explain_model_shap(model, X_test.sample(1000), model_type='lightgbm')

# Visualize feature importance
plot_shap_summary(shap_values, X_test.sample(1000))
```

## Usage Examples

See the Jupyter notebooks in the `notebooks/` directory for detailed examples:

- **05_explainability.ipynb**: Demonstrates SHAP-based model explanations and audit report generation
- **06_streaming_demo.ipynb**: Shows real-time transaction processing and scoring

## Testing

Run the test suite:

```bash
pytest tests/
```

## Configuration

Configuration files are located in the `configs/` directory. Modify `defaults.yaml` to adjust default parameters for data generation, model training, and evaluation.

## Development

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Mohammad Mahdi Mirrashid (1mmirrashid@gmail.com)

## Acknowledgments

This project uses the following open-source libraries:
- scikit-learn
- LightGBM, XGBoost, CatBoost
- SHAP for explainability
- SDV and CTGAN for synthetic data generation

