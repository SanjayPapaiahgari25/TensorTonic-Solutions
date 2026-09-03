import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    # Write code here
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    Wx = np.asarray(Wx)
    Wh = np.asarray(Wh)
    b = np.asarray(b)
    a = np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b
    
    return np.tanh(a)