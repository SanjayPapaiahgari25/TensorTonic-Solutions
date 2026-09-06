import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    p = np.asarray(p)
    y = np.asarray(y)

    dice = ((2*np.sum(p*y, axis=-1)) + eps) / (np.sum(p, axis=-1) + np.sum(y, axis=-1) + eps)
    return float(np.mean(1 - dice))