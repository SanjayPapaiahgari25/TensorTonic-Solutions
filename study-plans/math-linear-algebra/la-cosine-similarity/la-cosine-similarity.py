import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b)/(a_norm * b_norm)