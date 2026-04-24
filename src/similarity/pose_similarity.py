import numpy as np


def similarity_score(a, b):
    if a is None or b is None:
        return 0.0

    diff = a - b
    return 1.0 / (1.0 + np.linalg.norm(diff))
