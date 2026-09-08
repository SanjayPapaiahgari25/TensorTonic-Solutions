import math
import numpy as np
def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    L = math.sqrt(6/(fan_in + fan_out))

    W = np.asarray(W)
    return ((W*2*L) - L)
            