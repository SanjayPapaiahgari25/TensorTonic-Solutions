import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    n = Z1.shape[0]
    S = (Z1@Z2.T)/temperature
    p = []
    
    for i in range(n):
        p.append(np.exp(S[i, i]-np.max(S[i]))/np.sum(np.exp(S[i]-np.max(S[i])), axis=-1))
    
    L = -np.mean(np.log(p))
    
    return float(L)