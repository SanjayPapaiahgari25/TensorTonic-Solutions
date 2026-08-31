import numpy as np

def lbfgs_direction(grad: list, s_list: list, y_list: list) -> list:
    """
    Returns the L-BFGS descent direction from the stored history.
    """
    m = len(s_list)

    q = np.asarray(grad, dtype=float)
    s_list = [np.asarray(s, dtype=float) for s in s_list]
    y_list = [np.asarray(y, dtype=float) for y in y_list]

    rho = np.zeros(m)
    alpha = np.zeros(m)

    # First loop
    for i in range(m - 1, -1, -1):
        rho[i] = 1.0 / np.dot(y_list[i], s_list[i])
        alpha[i] = rho[i] * np.dot(s_list[i], q)

        q = q - alpha[i] * y_list[i]

    # Initial Hessian approximation H_0
    gamma = (
        np.dot(s_list[m - 1], y_list[m - 1])
        / np.dot(y_list[m - 1], y_list[m - 1])
    )

    r = gamma * q

    # Second loop
    for i in range(m):
        beta_i = rho[i] * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha[i] - beta_i)

    # Descent direction
    r = -r

    return r.tolist()