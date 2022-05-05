import numpy as np


def shortest_path(A, method="auto"):
    """ Calculate the shortest path distance matrix based on A.

    Args:
        A (np.ndarray): Adjacency matrix with dim (n, n).
        method (str, optional): Algorithm to use for shortest paths. ["auto" | "FW" | "D"].

    Returns:
        (np.ndarray): The commute time.
    """
    from scipy.sparse.csgraph import shortest_path
    assert isinstance(A, np.ndarray), "Input needs to be in (np.ndarray) type"
    C = shortest_path(A, directed=False)
    return C
