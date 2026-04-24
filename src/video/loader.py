import cv2


class VideoLoader:
    def __init__(self, video_path: str, skip_frames: int = 1):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.skip_frames = max(1, skip_frames)

        # metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        return self
    
    def __next__(self):
        for _ in range(self.skip_frames - 1):
            self.cap.grab()

        ret, frame = self.cap.read()

        if not ret:
            self.cap.release()
            raise StopIteration

        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def info(self):
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
        }