import math
import numpy as np

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    p_hat = np.minimum(1-eps, np.maximum(eps, y_pred))

    loss = -(y_true*np.log(p_hat) + (1-y_true)*np.log(1-p_hat))

    return loss.tolist()
    