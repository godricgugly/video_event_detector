# Video Event Detector (Pose-based)

Detects occurrences of a reference pose in a video using MediaPipe.

## Motivation & Goals

This project aims to provide a lightweight, pose-based event detection system for sports or movement analysis. By leveraging MediaPipe's pose estimation, it allows users to define key reference poses and automatically identify when they appear in a video. The goal is to create a robust MVP that can reliably detect events without manual annotation, providing timestamps for quick review.

## How It Works

### Reference Pose
Provide a short reference video (~2–3 seconds) of the pose you want to detect. The system averages landmarks across frames to build a stable reference.

### Pose Detection
MediaPipe extracts 33 landmarks per frame (x, y, z). Landmarks are normalized to be relative to the hips and scaled by shoulder width.

### Similarity & Event Detection
Each frame in the main video is compared to the reference pose. Temporal smoothing ensures events are only triggered when similarity is consistently above a threshold.

### Output
Prints timestamps where the reference pose appears.

### Usage
Self hosted app runs locally and offline, to avoid having to upload long videos.

## Stack
Python 3.10+, OpenCV, MediaPipe, NumPy, pytest