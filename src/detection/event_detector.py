
class EventDetector:
    def __init__(self, fps, threshold=0.5, duration_sec=0.1, min_fraction=0.7, cooldown_sec=5):
        self.fps = fps
        self.threshold = threshold
        self.required_frames = int(fps * duration_sec)
        self.min_fraction = min_fraction

        self.cooldown_frames = int(fps * cooldown_sec)

        self.buffer = []
        self.active = False
        self.start_frames = []
        self.frame_idx = 0

        self.cooldown_counter = 0

    def update(self, similarity):
        self.frame_idx += 1
        similarity = similarity or 0.0

        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        # Maintain sliding window
        self.buffer.append(similarity)
        if len(self.buffer) > self.required_frames:
            self.buffer.pop(0)

        if len(self.buffer) < self.required_frames:
            return False

        # Check fraction above threshold
        fraction_above = sum(s > self.threshold for s in self.buffer) / len(self.buffer)
        is_good = fraction_above >= self.min_fraction

        # Rising edge detection (ONLY trigger once)
        if is_good and not self.active:
            self.active = True

            # Backdate start to beginning of window (more accurate)
            start_frame = self.frame_idx - self.required_frames
            self.start_frames.append(start_frame)

            # Start cooldown immediately
            self.cooldown_counter = self.cooldown_frames

            return True

        elif not is_good and self.active:
            self.active = False
        return False

    def get_start_frames(self):
        return self.start_frames