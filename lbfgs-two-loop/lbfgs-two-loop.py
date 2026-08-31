import numpy as np

def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction from the stored history.
    """
    m = len(s_list)

    q = np.asarray(grad)
    s_list = np.asarray(s_list)
    y_list = np.asarray(y_list)

    rho = np.zeros(m)
    alpha = np.zeros(m)
    
    for i in range(m - 1, -1, -1):
        rho_i = (1.0 / np.dot(y_list[i], s_list[i])).item()
        alpha_i = (rho_i * np.dot(s_list[i], q)).item()
        rho[i]=rho_i
        alpha[i]=alpha_i
        q = q - alpha_i * y_list[i]

    r = (np.dot(s_list[m-1], y_list[m-1])/np.dot(y_list[m-1], y_list[m-1]))*q
    
    for i in range(m):
        r = r + s_list[i]*(alpha[i] - (rho[i]*np.dot(y_list[i], r)))
    r = -r
    return r.tolist()