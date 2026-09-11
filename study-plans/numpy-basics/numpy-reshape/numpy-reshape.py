import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    data = np.asarray(data)
    if operation == 'flatten':
        return np.astype(data.flatten(), np.float64)
    elif operation == 'transpose':
        return np.astype(data.T, np.float64)
    elif operation == 'add_batch':
        return np.astype(data.reshape(1, data.shape[0], data.shape[1]), np.float64)