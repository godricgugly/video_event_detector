import numpy as np
from src.similarity.pose_similarity import euclidean_distance, similarity_score


def test_similarity_basic():
    a = np.zeros(99)
    b = np.ones(99)

    dist = euclidean_distance(a, b)
    sim = similarity_score(a, b)

    assert dist > 0
    assert 0 < sim < 1