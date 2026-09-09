import numpy as np

def softmax(x):
    """Compute softmax values for a 1D vector x."""
    # Subtracting the max prevents numeric overflow
    e_x = np.exp(x - np.max(x)) 
    return e_x / np.sum(e_x)
    
def scaled_dot_product_attention(Q, K, V):
    """
    Returns: ndarray, the attention output softmax(Q @ K.T / sqrt(d_k)) @ V.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)
    d_k = Q.shape[1]

    scores = (Q @ K.T) / np.sqrt(d_k)

    # Apply softmax independently to each row
    attention_weights = np.apply_along_axis(softmax, axis=1, arr=scores)

    return attention_weights @ V