import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

def calculate_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    e_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    return np.sum((e_counts - a_counts) * np.log((e_counts + 1e-8) / (a_counts + 1e-8)))

def calculate_ks(expected, actual):
    return stats.ks_2samp(expected, actual)

def calculate_js(expected, actual, bins=50):
    e_hist, bin_edges = np.histogram(expected, bins=bins, density=True)
    a_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    return jensenshannon(e_hist + 1e-8, a_hist + 1e-8)
