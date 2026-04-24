import numpy as np
from src.similarity.pose_similarity import similarity_score


def test_similarity_basic():
    a = np.zeros(99)
    b = np.ones(99)
    
    sim = similarity_score(a, b)

    assert 0 < sim < 1