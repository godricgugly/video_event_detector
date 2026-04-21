import cv2
from src.pose.detector import PoseDetector
from src.pose.normalization import normalize_landmarks


def test_normalization_output_shape():
    detector = PoseDetector()

    frame = cv2.imread("tests/assets/frame.jpg")
    landmarks = detector.process(frame)

    if landmarks is None:
        detector.close()
        return

    vec = normalize_landmarks(landmarks)

    assert vec is None or vec.shape == (99,)

    detector.close()