import numpy as np

def projection_matrix(A: list) -> np.ndarray:
    """
    Returns the float64 projector onto the column space of A.
    """
    A = np.asarray(A)

    return A @ np.linalg.inv(A.T@A) @ A.T