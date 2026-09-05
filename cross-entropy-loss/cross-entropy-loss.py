import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    n = len(y_true)
    bce = 0.0
    for i in range(n):
        bce += np.log(y_pred[i][y_true[i]])
    bce = bce / n
    # print(bce)
    return -float(bce)
    