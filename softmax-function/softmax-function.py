import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim == 1:
        m = np.max(x)
    elif x.ndim == 2:
        m = np.max(x, axis=1, keepdims=True)
    x = x - m
    if x.ndim == 1:
        return np.exp(x)/np.sum(np.exp(x))
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)