import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p)
    q = np.asarray(q)
    q = q + eps

    kl_d = np.where(p!=0, p*np.log(p/q), 0)
    return float(np.sum(kl_d))