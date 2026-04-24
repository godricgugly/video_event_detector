import cv2
import numpy as np
import mediapipe as mp


class PoseDetector:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        """
        Input: BGR frame
        Output: np.array shape (33, 4) or None
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)

        if not result.pose_landmarks:
            return None

        # VECTORISED OUTPUT
        landmarks = np.array(
            [[
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ] for lm in result.pose_landmarks.landmark],
            dtype=np.float32
        )

        return landmarks

    def close(self):
        self.pose.close()