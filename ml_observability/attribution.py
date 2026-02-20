import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _align_pairwise(a, b):
    """Align two arrays to the same length for pairwise comparison."""
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("Inputs must contain at least one value.")
    return np.asarray(a)[:n], np.asarray(b)[:n]


def confidence_attribution(train_df, prod_df, train_probs, prod_probs, features):
    """Estimate which feature shifts are most correlated with confidence shift."""
    train_probs, prod_probs = _align_pairwise(train_probs, prod_probs)
    conf_shift = prod_probs - train_probs

    results = []
    for feature in features:
        train_vals, prod_vals = _align_pairwise(train_df[feature].values, prod_df[feature].values)
        shift = prod_vals - train_vals

        corr, _ = spearmanr(np.abs(shift), np.abs(conf_shift[: len(shift)]))
        results.append(
            {
                "feature": feature,
                "confidence_correlation": float(corr) if corr is not None else np.nan,
            }
        )

    return pd.DataFrame(results).sort_values("confidence_correlation", ascending=False)
