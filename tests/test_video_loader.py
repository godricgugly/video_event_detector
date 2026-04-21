import os
import pytest
from src.video.loader import VideoLoader

TEST_VIDEO = "data/raw/session.mp4"


def test_video_loader_opens():
    loader = VideoLoader(TEST_VIDEO)
    info = loader.info()

    assert info["fps"] > 0
    assert info["frame_count"] > 0
    assert info["width"] > 0
    assert info["height"] > 0

    loader.release()


def test_video_loader_invalid_path():
    with pytest.raises(ValueError):
        VideoLoader("non_existent.mp4")


def test_video_loader_iteration():
    loader = VideoLoader(TEST_VIDEO)

    frames = []
    for i, frame in enumerate(loader):
        frames.append(frame)
        if i > 5:
            break

    assert len(frames) > 0

    loader.release()