import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    z = (values**3) + (2*values)
    y = torch.sum(z)
    y.backward()
    return values.grad.tolist()