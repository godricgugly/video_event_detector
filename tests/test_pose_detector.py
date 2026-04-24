import numpy as np
import cv2
from unittest.mock import MagicMock
from src.pose.detector import PoseDetector


def test_pose_detector_runs():
    detector = PoseDetector()

    frame = cv2.imread("tests/assets/frame.jpg")
    landmarks = detector.process(frame)

    assert landmarks is None or isinstance(landmarks, np.ndarray)

    if landmarks is not None:
        assert landmarks.shape == (33, 4)

    detector.close()


def test_pose_detector_output_format():
    detector = PoseDetector()

    frame = cv2.imread("tests/assets/frame.jpg")
    landmarks = detector.process(frame)

    if landmarks is not None:
        assert isinstance(landmarks, np.ndarray)
        assert landmarks.shape == (33, 4)

        assert np.all(landmarks[:, :3] >= -1.0)
        assert np.all(landmarks[:, :3] <= 1.0)
        assert np.all(landmarks[:, 3] >= 0.0)

    detector.close()


def test_pose_detector_mocked():
    detector = PoseDetector()

    fake_result = MagicMock()

    fake_landmarks = []
    for _ in range(33):
        lm = MagicMock()
        lm.x = 0.1
        lm.y = 0.2
        lm.z = 0.3
        lm.visibility = 0.9
        fake_landmarks.append(lm)

    fake_result.pose_landmarks.landmark = fake_landmarks

    detector.pose.process = MagicMock(return_value=fake_result)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    landmarks = detector.process(frame)

    assert isinstance(landmarks, np.ndarray)
    assert landmarks.shape == (33, 4)

    # check values
    assert np.isclose(landmarks[0, 0], 0.1)
    assert np.isclose(landmarks[0, 1], 0.2)
    assert np.isclose(landmarks[0, 2], 0.3)
    assert np.isclose(landmarks[0, 3], 0.9)

    detector.close()