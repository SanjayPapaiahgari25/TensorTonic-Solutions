import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u = np.asarray(u).reshape(-1,1)
    v = np.asarray(v).reshape(-1, 1)
    return u @ v.T