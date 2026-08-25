import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.asarray(v, dtype=np.float64)
    l1_norm = np.sum(np.abs(v))
    l2_norm = np.sqrt(np.sum(v**2))
    l_inf_norm = np.max(np.abs(v))
    return np.array([l1_norm, l2_norm, l_inf_norm], dtype=np.float64)