import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    # Write code here
    x = np.asarray(x)
    lambda_selu = 1.0507
    alpha = 1.6733

    return np.where(x>0, lambda_selu*x, lambda_selu*alpha*(np.exp(x) - 1))