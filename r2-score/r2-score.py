import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mu = np.mean(y_true)
    if len(np.unique(y_true)) == 1:
        if (y_true == y_pred).all():
            return 1.0
        else:
            return 0.0
    r_sq = 1 - (np.sum((y_true-y_pred) ** 2)/np.sum((y_true - mu)**2))

    return float(r_sq)