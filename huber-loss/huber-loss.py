import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    error = y_true-y_pred
    loss = np.mean(np.where(np.abs(error) <= delta, 0.5 * error**2, delta*(np.abs(error) - (0.5*delta))))
    return float(loss)