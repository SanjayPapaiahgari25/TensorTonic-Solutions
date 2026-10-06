import numpy as np


def relu(x):
    return [np.where(x>0, x, 0), np.where(x>0, 1, 0)]

def leaky_relu(x):
    return [np.where(x>0, x, 0.01*x), np.where(x>0, 1, 0.01)]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
  
def gelu(x):
    u = np.dot(np.sqrt(2/np.pi), (x + np.dot(0.044715, x**3)))
    t = tanh(u)
    gelu_x = 0.5*np.dot(x, 1+t)

    u_dash =  np.dot(np.sqrt(2/np.pi), (1 + 3*np.dot(0.044715, x**2)))

    gelu_dash_x = 0.5*(1+t) + (0.5*np.dot(x, np.dot((1-t**2), u_dash)))
    return [gelu_x, gelu_dash_x]

def swish(x):
    sigma = sigmoid(x)
    return [np.dot(x, sigma), sigma + np.dot(x, sigma*(1-sigma))]

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative.
    """
    x = np.asarray(x)

    if activation == 'relu':
        return relu(x)
    elif activation == 'leaky_relu':
        return leaky_relu(x)
    elif activation == 'sigmoid':
        sigma = sigmoid(x)
        return [sigma, sigma*(1-sigma)]
    elif activation == 'tanh':
        tanhx = tanh(x)
        return [tanhx, 1 - (tanhx**2)]
    elif activation == 'swish':
        return swish(x)
    elif activation == 'gelu':
        return gelu(x)
        
    
