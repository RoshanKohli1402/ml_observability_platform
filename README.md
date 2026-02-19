# ML Observability & Drift Intelligence Platform

A lightweight Python platform for **monitoring data drift, model behavior drift, and decision-logic changes** in deployed machine learning systems.

This library helps detect **silent model failures** caused by changes in production data—without requiring immediate access to labels.

---

## Why This Exists

Machine learning models often degrade in production due to:
- Data distribution shifts
- Segment-specific changes
- Delayed or missing labels
- Changes in how models use features over time

Traditional evaluation pipelines fail to detect these issues early.

This platform provides **ML observability** by continuously monitoring:
- Feature-level data drift
- Prediction confidence drift
- Root-cause features driving degradation
- Explainability (decision-logic) drift using SHAP

---

## Key Capabilities

- **Global Feature Drift Detection**
  - PSI, KS test, Jensen–Shannon distance
- **Model Behavior Monitoring**
  - Prediction confidence drift (label-independent)
- **Root-Cause Attribution**
  - Identifies top 3–5 features driving prediction changes
- **Explainability Drift (Optional)**
  - SHAP-based detection of decision-logic changes
- **Plug-and-Play API**
  - Works with Pandas DataFrames
  - No retraining required

---

## Installation

Clone the repository and install locally:

```bash
git clone <your-repo-url>
cd ml-observability-platform
pip install -e .
