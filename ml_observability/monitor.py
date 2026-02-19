import pandas as pd
from .drift_metrics import calculate_psi, calculate_ks, calculate_js

class DriftMonitor:
    def __init__(self, segment_col=None):
        self.segment_col = segment_col
        self.train_df = None
        self.model = None

    def fit_baseline(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()

    def attach_model(self, model):
        self.model = model

    def monitor(self, prod_df: pd.DataFrame):
        results = []

        for feature in self.train_df.columns:
            if feature == self.segment_col:
                continue

            psi = calculate_psi(self.train_df[feature], prod_df[feature])
            ks_stat, ks_p = calculate_ks(self.train_df[feature], prod_df[feature])
            js = calculate_js(self.train_df[feature], prod_df[feature])

            results.append({
                "feature": feature,
                "PSI": psi,
                "KS_pvalue": ks_p,
                "JS_distance": js
            })

        drift_df = pd.DataFrame(results).sort_values("PSI", ascending=False)

        return {"global_drift": drift_df}
