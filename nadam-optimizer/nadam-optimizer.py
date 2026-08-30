import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Write code here
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)

    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*(grad**2)
    m_hat = beta1*m + (1-beta1)*grad
    w = w - (lr*(m_hat/(np.sqrt(v)+eps)))

    return {"new_w": w, "new_m": m, "new_v": v}