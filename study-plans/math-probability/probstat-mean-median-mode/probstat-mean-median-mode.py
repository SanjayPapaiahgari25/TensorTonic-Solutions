import numpy as np
from scipy import stats

def mean_median_mode(x: list) -> dict:
    """
    Returns mean, median, and mode as Python floats in a dictionary.
    """
    x = np.asarray(x)
    return dict({"mean": float(np.mean(x)), "median": float(np.median(x)), "mode": float(stats.mode(x, keepdims=True).mode[0])})