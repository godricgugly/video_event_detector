import cv2
from src.pose.detector import PoseDetector
from unittest.mock import MagicMock


def test_pose_detector_runs():
    detector = PoseDetector()

    frame = cv2.imread("tests/assets/frame.jpg")
    landmarks = detector.process(frame)

    # It may or may not detect a pose — both are valid
    assert landmarks is None or isinstance(landmarks, list)

    detector.close()


def test_pose_detector_output_format():
    detector = PoseDetector()

    frame = cv2.imread("tests/assets/frame.jpg")
    landmarks = detector.process(frame)

    if landmarks is not None:
        assert isinstance(landmarks, list)
        assert "x" in landmarks[0]
        assert "y" in landmarks[0]
        assert "z" in landmarks[0]
        assert "visibility" in landmarks[0]

    detector.close()

    from unittest.mock import MagicMock
from src.pose.detector import PoseDetector


def test_pose_detector_mocked():
    detector = PoseDetector()

    fake_result = MagicMock()
    fake_landmark = MagicMock(x=0.1, y=0.2, z=0.3, visibility=0.9)
    fake_result.pose_landmarks.landmark = [fake_landmark] * 33

    detector.pose.process = MagicMock(return_value=fake_result)

    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    landmarks = detector.process(frame)

    assert len(landmarks) == 33
    assert landmarks[0]["x"] == 0.1

    detector.close()