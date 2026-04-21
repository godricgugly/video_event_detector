import cv2
from src.pose.detector import PoseDetector
from src.pose.normalization import normalize_landmarks
from src.reference.reference_builder import build_reference_pose


def test_reference_pose_builds():
    detector = PoseDetector()

    frame = cv2.imread("tests/assets/frame.jpg")
    landmarks = detector.process(frame)

    if landmarks is None:
        detector.close()
        return

    # normalize before building reference
    norm1 = normalize_landmarks(landmarks)
    norm2 = normalize_landmarks(landmarks)

    if norm1 is None or norm2 is None:
        detector.close()
        return

    ref = build_reference_pose([norm1, norm2])

    assert ref is not None
    assert ref.shape == (99,)

    detector.close()