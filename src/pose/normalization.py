import numpy as np


# MediaPipe landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def normalize_landmarks(landmarks):
    """
    Input: list of 33 landmarks (dicts with x, y, z)
    Output: flattened normalized vector (99,)
    """

    if landmarks is None:
        return None

    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])

    # --- 1. center (hip midpoint) ---
    hip_center = (pts[LEFT_HIP] + pts[RIGHT_HIP]) / 2.0
    pts = pts - hip_center

    # --- 2. scale (shoulder width) ---
    shoulder_dist = np.linalg.norm(
        pts[LEFT_SHOULDER] - pts[RIGHT_SHOULDER]
    )

    if shoulder_dist < 1e-6:
        return None  # invalid pose

    pts = pts / shoulder_dist

    return pts.flatten()