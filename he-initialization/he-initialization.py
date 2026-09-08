import math
import numpy as np

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    L = math.sqrt(6/fan_in)
    W = np.asarray(W)

    return ((W*2*L) - L)