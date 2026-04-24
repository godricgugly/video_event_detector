import numpy as np

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def normalize_landmarks(landmarks):
    """
    Input: np.array shape (33, 4) -> [x, y, z, visibility]
    Output: flattened normalized vector (99,)
    """

    if landmarks is None:
        return None

    # keep only xyz (drop visibility for geometry)
    pts = landmarks[:, :3].astype(np.float32)

    # --- 1. center (hip midpoint) ---
    hip_center = (pts[LEFT_HIP] + pts[RIGHT_HIP]) * 0.5
    pts -= hip_center

    # --- 2. scale (shoulder width) ---
    shoulder_vec = pts[LEFT_SHOULDER] - pts[RIGHT_SHOULDER]
    shoulder_dist = np.linalg.norm(shoulder_vec)

    if shoulder_dist < 1e-6:
        return None

    pts /= shoulder_dist

    return pts.reshape(-1)