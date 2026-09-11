import numpy as np

def extract_subarray(arr, row_start, row_stop, col_start, col_stop):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.asarray(arr)
    return np.astype(arr[row_start:row_stop, col_start: col_stop], np.float64)
