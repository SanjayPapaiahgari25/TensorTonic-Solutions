import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    n_f, n_r = len(fake_scores), len(real_scores)

    real_score = (1/n_r)*sum(real_scores)
    fake_score = (1/n_f)*sum(fake_scores)
    return fake_score - real_score