import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    vectors = np.asarray(vectors)
    coefficients = np.asarray(coefficients)
    return coefficients.T @ vectors

    