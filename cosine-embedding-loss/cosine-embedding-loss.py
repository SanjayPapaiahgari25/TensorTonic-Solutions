import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    x1_norm = np.linalg.norm(x1)
    x2_norm = np.linalg.norm(x2)
    eps = 1e-8
    cosine = np.dot(x1, x2)/(x1_norm * x2_norm + eps)
    return np.mean(np.where(label==1, 1-cosine, np.maximum(0, cosine - margin)))