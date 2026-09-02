import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x)
    return x*(1/(1+np.exp(-x)))