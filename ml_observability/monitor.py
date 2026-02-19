import pandas as pd

from .drift_metrics import (
    calculate_psi,
    calculate_ks,
    calculate_js
)

from .attribution import confidence_attribution


class DriftMonitor:
    """
    Core interface for ML Observability & Drift Intelligence.

    Usage:
        monitor = DriftMonitor(segment_col=None)
        monitor.fit_baseline(train_df)
        monitor.attach_model(model)        # optional
        report = monitor.monitor(prod_df)
    """

    def __init__(self, segment_col: str = None):
        self.segment_col = segment_col
        self.train_df = None
        self.model = None

    def fit_baseline(self, train_df: pd.DataFrame):
        """
        Register baseline (training) data.
        """
        self.train_df = train_df.copy()

    def attach_model(self, model):
        """
        Attach a trained model (must support predict_proba).
        Enables confidence drift & attribution.
        """
        self.model = model

    def monitor(self, prod_df: pd.DataFrame):
        """
        Run drift detection on production data.

        Returns:
            report (dict) with:
                - global_drift
                - confidence_drift (optional)
                - root_causes (optional)
        """

        if self.train_df is None:
            raise ValueError("Baseline not set. Call fit_baseline() first.")

        results = []

        # -------------------------------
        # GLOBAL FEATURE DRIFT
        # -------------------------------
        for feature in self.train_df.columns:
            if feature == self.segment_col:
                continue

            psi = calculate_psi(
                self.train_df[feature],
                prod_df[feature]
            )

            _, ks_p = calculate_ks(
                self.train_df[feature],
                prod_df[feature]
            )

            js = calculate_js(
                self.train_df[feature],
                prod_df[feature]
            )

            results.append({
                "feature": feature,
                "PSI": psi,
                "KS_pvalue": ks_p,
                "JS_distance": js
            })

        drift_df = (
            pd.DataFrame(results)
            .sort_values("PSI", ascending=False)
            .reset_index(drop=True)
        )

        report = {
            "global_drift": drift_df
        }

        # -------------------------------
        # OPTIONAL: MODEL-BASED INTELLIGENCE
        # -------------------------------
        if self.model is not None:
            # Prediction confidence drift
            train_probs = self.model.predict_proba(self.train_df)[:, 1]
            prod_probs = self.model.predict_proba(prod_df)[:, 1]

            report["confidence_drift"] = {
                "PSI": calculate_psi(train_probs, prod_probs)
            }

            # Root-cause attribution (top drifting features)
            top_features = drift_df.head(5)["feature"].tolist()

            report["root_causes"] = confidence_attribution(
                train_df=self.train_df[top_features],
                prod_df=prod_df[top_features],
                train_probs=train_probs,
                prod_probs=prod_probs,
                features=top_features
            )

        return report
