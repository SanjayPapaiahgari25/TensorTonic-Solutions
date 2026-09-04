from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x = np.asarray(sorted(x))
    count_tuple = np.unique(x, return_counts=True)
    max_index = np.argmax(count_tuple[1])
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "mode": float(x[max_index])}