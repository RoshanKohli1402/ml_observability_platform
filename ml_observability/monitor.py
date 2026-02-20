import pandas as pd

from .drift_metrics import (
    calculate_psi,
    calculate_ks,
    calculate_js,
)
from .attribution import confidence_attribution


class DriftMonitor:
    """
    Core interface for ML observability and drift intelligence.
    """

    def __init__(self, segment_col: str = None):
        self.segment_col = segment_col
        self.train_df = None
        self.model = None

    def fit_baseline(self, train_df: pd.DataFrame):
        """Register baseline (training) data."""
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a pandas DataFrame.")
        if train_df.empty:
            raise ValueError("train_df must not be empty.")
        self.train_df = train_df.copy()

    def attach_model(self, model):
        """Attach a trained model supporting predict_proba."""
        if not hasattr(model, "predict_proba"):
            raise TypeError("model must expose a predict_proba method.")
        self.model = model

    def _validate_prod_input(self, prod_df: pd.DataFrame):
        if not isinstance(prod_df, pd.DataFrame):
            raise TypeError("prod_df must be a pandas DataFrame.")
        if prod_df.empty:
            raise ValueError("prod_df must not be empty.")

        missing_cols = [c for c in self.train_df.columns if c not in prod_df.columns]
        if missing_cols:
            raise ValueError(f"prod_df is missing baseline columns: {missing_cols}")

    def monitor(self, prod_df: pd.DataFrame):
        """
        Run drift detection on production data.

        Returns:
            dict with:
                - global_drift
                - confidence_drift (optional)
                - root_causes (optional)
        """
        if self.train_df is None:
            raise ValueError("Baseline not set. Call fit_baseline() first.")

        self._validate_prod_input(prod_df)

        results = []
        for feature in self.train_df.columns:
            if feature == self.segment_col:
                continue

            psi = calculate_psi(self.train_df[feature], prod_df[feature])
            _, ks_p = calculate_ks(self.train_df[feature], prod_df[feature])
            js = calculate_js(self.train_df[feature], prod_df[feature])

            results.append(
                {
                    "feature": feature,
                    "PSI": psi,
                    "KS_pvalue": ks_p,
                    "JS_distance": js,
                }
            )

        drift_df = pd.DataFrame(results).sort_values("PSI", ascending=False).reset_index(drop=True)

        report = {"global_drift": drift_df}

        if self.model is not None:
            train_probs = self.model.predict_proba(self.train_df)[:, 1]
            prod_probs = self.model.predict_proba(prod_df[self.train_df.columns])[:, 1]

            report["confidence_drift"] = {
                "PSI": calculate_psi(train_probs, prod_probs)
            }

            top_features = drift_df.head(5)["feature"].tolist()
            report["root_causes"] = confidence_attribution(
                train_df=self.train_df[top_features],
                prod_df=prod_df[top_features],
                train_probs=train_probs,
                prod_probs=prod_probs,
                features=top_features,
            )

        return report
