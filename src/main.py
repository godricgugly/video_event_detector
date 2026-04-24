import time

start_total = time.perf_counter()

#############################################################
from datetime import timedelta
import cv2
from src.video.loader import VideoLoader
from src.pose.detector import PoseDetector
from src.pose.normalization import normalize_landmarks
from src.reference.reference_builder import build_reference_pose
from src.similarity.pose_similarity import similarity_score
from src.detection.event_detector import EventDetector

# -----------------------------
# USER PARAMETERS
# -----------------------------
reference_video_path = "data/references/ref_pose.mp4"
main_video_path = "data/raw/session.mp4"
skip_frames = 6 # Reduce for more accuracy (slower)
similarity_threshold = 0.4 
duration_sec = 0.1
cooldown_sec = 3
model_complexity = 0  # 0 = fast, 1 = balanced, 2 = accurate
# -----------------------------



# --- STEP 1
start = time.perf_counter()

# --- STEP 1: Build reference pose
ref_loader = VideoLoader(reference_video_path)
pose_detector = PoseDetector(model_complexity=model_complexity)

ref_landmarks_list = []

for frame in ref_loader:
    lm = pose_detector.process(frame)
    if lm:
        norm = normalize_landmarks(lm)
        ref_landmarks_list.append(norm)

pose_detector.close()
ref_loader.release()

if not ref_landmarks_list:
    raise ValueError("No pose detected in reference video!")

# Average landmarks over all frames in reference clip
reference_pose = build_reference_pose(ref_landmarks_list)

###
end = time.perf_counter()
print(f"Step 1 (reference build): {end - start:.3f}s")

# --- STEP 2
start = time.perf_counter()

# --- STEP 2: Search main video
loader = VideoLoader(main_video_path, skip_frames=skip_frames)
info = loader.info()

pose_detector = PoseDetector(model_complexity=model_complexity)
detector = EventDetector(info["fps"], threshold=similarity_threshold, duration_sec=duration_sec, cooldown_sec=cooldown_sec)

for i, frame in enumerate(loader):
    lm = pose_detector.process(frame)
    if not lm:
        continue
    norm = normalize_landmarks(lm)
    sim = similarity_score(norm, reference_pose)
    detector.update(sim)

pose_detector.close()
loader.release()

###
end = time.perf_counter()
print(f"Step 2 (main video scan): {end - start:.3f}s")


# --- STEP 3
start = time.perf_counter()

# --- STEP 3: Output timestamps
start_frames = detector.get_start_frames()
start_timestamps = [(f * skip_frames) / info["fps"] for f in start_frames]

formatted_times = [
    str(timedelta(seconds=round(t))) for t in start_timestamps
]

print("Event starts at (hh:mm:ss):", formatted_times)

###
end = time.perf_counter()
print(f"Step 3 (post-processing): {end - start:.3f}s")

###

end_total = time.perf_counter()
print(f"Total runtime: {end_total - start_total:.3f}s")