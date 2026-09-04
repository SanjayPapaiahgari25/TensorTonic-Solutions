import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x)
    n = len(x)
    erf = np.vectorize(lambda a: math.erf(a/math.sqrt(2.0)))
    x = 0.5*x*(1+erf(x))
    return x