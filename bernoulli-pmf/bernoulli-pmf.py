import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    pmf = np.array([1-p if i==0 else p for i in x])
    return {"pmf": pmf, "mean": p, "variance": p*(1-p)}