import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    L = math.sqrt(6/fan_in)
    m, n = len(W), len(W[0])
    for i in range(m):
        for j in range(n):
            W[i][j] = (W[i][j]*2*L) - L

    return W