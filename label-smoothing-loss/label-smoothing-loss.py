import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    # Write code here
    n = len(predictions)
    K = n
    loss = 0.0
    for i in range(n):
        if i == target:
            q_i = (1 - epsilon) + (epsilon/K)
        else:
            q_i = (epsilon/K)
        loss += (q_i*math.log(predictions[i]))
    return -loss