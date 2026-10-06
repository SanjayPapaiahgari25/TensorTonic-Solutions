import numpy as np


def relu(x):
    if x > 0:
        return [x, 1]
    return [0, 0]

def leaky_relu(x):
    if x > 0:
        return [x, 1]
    return [0.01*x, 0.01]

def sigmoid(x):
    sigma = 1/(1 + np.exp(-x))
    return sigma

def tanh(x):
    tanhx = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return tanhx
  
def gelu(x):
    t = tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))
    gelu_x = 0.5*x*(1+t)

    u_dash = np.sqrt(2/np.pi) * (1 + 3*0.044715*x**2)
    gelu_dash_x = 0.5*(1+t + (x * (1-t**2) * u_dash))
    
    return [gelu_x, gelu_dash_x]

def swish(x):
    sigma = sigmoid(x)
    return [x * sigma, sigma + x*sigma*(1-sigma)]

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative.
    """

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
        
    
