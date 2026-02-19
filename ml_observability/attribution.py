import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def confidence_attribution(train_df, prod_df, train_probs, prod_probs, features):
    results = []

    for feature in features:
        shift = prod_df[feature].values - train_df[feature].values
        conf_shift = prod_probs - train_probs

        corr, _ = spearmanr(np.abs(shift), np.abs(conf_shift))

        results.append({
            "feature": feature,
            "confidence_correlation": corr
        })

    return (
        pd.DataFrame(results)
        .sort_values("confidence_correlation", ascending=False)
    )
