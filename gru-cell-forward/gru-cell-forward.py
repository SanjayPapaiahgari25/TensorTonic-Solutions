import numpy as np

def _sigmoid(x):
    return 1/(1+np.exp(-x))

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    # Write code here
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)
    params = {k: np.asarray(v) for k, v in params.items()}
    z_t = _sigmoid(x@params["Wz"] + h_prev@params["Uz"] + params["bz"])
    r_t = _sigmoid(x@params["Wr"] + h_prev@params["Ur"] + params["br"])
    print(z_t.shape)
    print(r_t.shape)
    h_candidate = np.tanh(x@params["Wh"] + (r_t*h_prev)@params["Uh"] + params["bh"])
    print(h_candidate.shape)
    h = ((1-z_t)*h_prev) + z_t*h_candidate
    return h