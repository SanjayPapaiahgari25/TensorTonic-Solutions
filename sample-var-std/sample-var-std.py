import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.asarray(x)
    mean = np.mean(x)
    n = len(x)

    variance = (1/(n-1))*np.sum((x-mean)**2)
    std = variance ** 0.5

    return {"variance": float(variance), "standard_deviation": float(std)}