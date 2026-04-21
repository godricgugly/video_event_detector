import numpy as np


def build_reference_pose(vectors):
    """
    Input: list of (99,) normalized pose vectors
    Output: averaged reference vector (99,)
    """

    vectors = [v for v in vectors if v is not None]

    if len(vectors) == 0:
        return None

    return np.mean(vectors, axis=0)