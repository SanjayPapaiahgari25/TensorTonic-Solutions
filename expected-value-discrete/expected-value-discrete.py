import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    n = len(x)
    x = np.asarray(x)
    p = np.asarray(p)

    return float(np.dot(x,p))