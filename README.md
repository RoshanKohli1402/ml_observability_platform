# ML Observability & Drift Intelligence Platform

A lightweight Python library for monitoring **data drift**, **model behavior drift**, and optional **explainability drift** in deployed ML systems.

## What it does

- Global feature drift with PSI, KS test p-value, and Jensen–Shannon distance.
- Optional model confidence drift (label-independent).
- Root-cause feature attribution for confidence movement.
- Optional SHAP-based explainability drift.

## Installation

```bash
pip install -e .
```

## Quickstart

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ml_observability import DriftMonitor

# Data
raw = load_breast_cancer()
X = pd.DataFrame(raw.data, columns=raw.feature_names)
y = raw.target

train_X, prod_X = train_test_split(X, test_size=0.4, random_state=42)

# Inject controlled drift
prod_X = prod_X.copy()
prod_X["mean radius"] *= 1.25
prod_X["mean texture"] *= 1.15

# Train any model with predict_proba
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(train_X, y[: len(train_X)])

# Monitor
monitor = DriftMonitor()
monitor.fit_baseline(train_X)
monitor.attach_model(model)
report = monitor.monitor(prod_X)

print(report["global_drift"].head())
print(report["confidence_drift"])
print(report["root_causes"].head())
```

See `examples/quickstart.py` for a runnable script.

## API summary

### `DriftMonitor.fit_baseline(train_df)`
Registers baseline data. `train_df` must be a non-empty pandas DataFrame.

### `DriftMonitor.attach_model(model)`
Attaches a model with `predict_proba`.

### `DriftMonitor.monitor(prod_df)`
Computes drift report for `prod_df` and validates schema compatibility.

## Development

Run checks:

```bash
pytest -q
python examples/quickstart.py
```
