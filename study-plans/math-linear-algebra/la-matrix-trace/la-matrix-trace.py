import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    A = np.asarray(A)
    trace = 0.0
    m = A.shape[0]
    for i in range(m):
        trace += A[i, i]
    return trace
    