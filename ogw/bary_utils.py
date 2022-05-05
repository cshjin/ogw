import numpy as np


def laplacian(A):
    """ Get graph Laplacian matrix

    Args:
        A (np.ndarray): Adjacency matrix

    Returns:
        (np.ndarray): Laplacian matrix
    """
    return np.diag(A.sum(1)) - A

