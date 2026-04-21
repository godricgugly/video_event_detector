import numpy as np


def euclidean_distance(a, b):
    if a is None or b is None:
        return None

    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(a - b)


def similarity_score(a, b):
    """
    Converts distance → bounded similarity in (0, 1]
    """
    dist = euclidean_distance(a, b)

    if dist is None:
        return 0.0

    return 1.0 / (1.0 + dist)