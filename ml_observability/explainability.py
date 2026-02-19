import pandas as pd
import shap
from .drift_metrics import calculate_psi

def shap_drift(model, scaler, train_df, prod_df):
    X_train_scaled = scaler.transform(train_df)
    X_prod_scaled = scaler.transform(prod_df)

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_train = explainer.shap_values(X_train_scaled)
    shap_prod = explainer.shap_values(X_prod_scaled)

    results = []

    for i, feature in enumerate(train_df.columns):
        psi = calculate_psi(shap_train[:, i], shap_prod[:, i])
        results.append({
            "feature": feature,
            "SHAP_PSI": psi
        })

    return (
        pd.DataFrame(results)
        .sort_values("SHAP_PSI", ascending=False)
    )
