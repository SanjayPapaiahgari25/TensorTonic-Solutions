import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    # Write code here
    n = len(targets)
    bin_focal_loss = 0.0
    for i in range(n):
        if targets[i] == 1:
            p_t = predictions[i]
        else:
            p_t = 1 - predictions[i]
        bin_focal_loss += -alpha * math.pow((1-p_t), gamma)*math.log(p_t)
    return bin_focal_loss/n
            
        