import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.shape != b.shape:
        raise ValueError("Input shapes must match")
    
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
    
    return np.dot(a, b)/(a_norm * b_norm)