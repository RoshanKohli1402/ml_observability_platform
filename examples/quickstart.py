from pathlib import Path
import sys

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Allow running this script directly from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_observability import DriftMonitor
from ml_observability.explainability import shap_drift


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    train_X, prod_X = train_test_split(X, test_size=0.4, random_state=42)

    # Inject controlled drift into production
    prod_X = prod_X.copy()
    prod_X["mean radius"] *= 1.25
    prod_X["mean texture"] *= 1.15

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(train_X, y.loc[train_X.index])

    monitor = DriftMonitor()
    monitor.fit_baseline(train_X)
    monitor.attach_model(model)

    report = monitor.monitor(prod_X)
    print("Top global drift features")
    print(report["global_drift"].head(10))
    print("\nConfidence drift")
    print(report["confidence_drift"])
    print("\nRoot causes")
    print(report["root_causes"])

    shap_report = shap_drift(
        model=model.named_steps["clf"],
        scaler=model.named_steps["scaler"],
        train_df=train_X,
        prod_df=prod_X,
    )
    print("\nTop SHAP drift features")
    print(shap_report.head(10))


if __name__ == "__main__":
    main()
