import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n = y_pred.shape[0]

    return 1/n*(np.sum((y_pred - y_true)**2))