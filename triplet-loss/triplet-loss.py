import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    d_a_p = np.sum((anchor - positive)**2, axis=-1)
    d_a_n = np.sum((anchor - negative)**2, axis=-1)
    return float(np.mean(np.maximum(0, (d_a_p - d_a_n) + margin)))