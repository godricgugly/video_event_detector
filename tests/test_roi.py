import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.video.roi import select_roi


# Helper to load test frame
def load_test_frame():
    frame = cv2.imread("tests/assets/frame.jpg")
    assert frame is not None, "Test frame not found"
    return frame


@patch("cv2.VideoCapture")
@patch("cv2.selectROI")
def test_select_roi_returns_scaled_coordinates(mock_select, mock_cap):
    frame = load_test_frame()
    h, w = frame.shape[:2]

    # Fake VideoCapture
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (True, frame)
    mock_cap.return_value = mock_instance

    # Simulate ROI on resized image
    mock_select.return_value = (50, 60, 100, 120)

    roi = select_roi("fake_path.mp4", max_width=200, max_height=200)

    assert roi is not None
    x, y, rw, rh = roi

    # Should scale UP (since resized smaller)
    assert x >= 50
    assert y >= 60
    assert rw >= 100
    assert rh >= 120


@patch("cv2.VideoCapture")
@patch("cv2.selectROI")
def test_select_roi_no_resize(mock_select, mock_cap):
    frame = load_test_frame()

    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (True, frame)
    mock_cap.return_value = mock_instance

    # ROI directly on original frame
    mock_select.return_value = (10, 20, 30, 40)

    roi = select_roi("fake_path.mp4", max_width=9999, max_height=9999)

    assert roi == (10, 20, 30, 40)


@patch("cv2.VideoCapture")
@patch("cv2.selectROI")
def test_select_roi_returns_none_on_empty_selection(mock_select, mock_cap):
    frame = load_test_frame()

    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (True, frame)
    mock_cap.return_value = mock_instance

    mock_select.return_value = (0, 0, 0, 0)

    roi = select_roi("fake_path.mp4")

    assert roi is None


@patch("cv2.VideoCapture")
def test_select_roi_video_not_open(mock_cap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = False
    mock_cap.return_value = mock_instance

    with pytest.raises(ValueError, match="Could not open video"):
        select_roi("bad_path.mp4")


@patch("cv2.VideoCapture")
def test_select_roi_frame_read_fail(mock_cap):
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (False, None)
    mock_cap.return_value = mock_instance

    with pytest.raises(ValueError, match="Could not read frame"):
        select_roi("fake_path.mp4")