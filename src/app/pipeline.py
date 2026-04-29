from datetime import timedelta
from threading import Event

from src.video.loader import VideoLoader
from src.pose.detector import PoseDetector
from src.pose.normalization import normalize_landmarks
from src.reference.reference_builder import build_reference_pose
from src.similarity.pose_similarity import similarity_score
from src.detection.event_detector import EventDetector


def run_detection(
    reference_video_path: str,
    main_video_path: str,
    skip_frames: int = 4,
    similarity_threshold: float = 0.25,
    duration_sec: float = 0.1,
    cooldown_sec: float = 7,
    model_complexity: int = 0,
    roi=None,
    progress_callback=None,
    stop_event: Event = None,
):
    # -----------------------------
    # PREP: compute total work
    # -----------------------------
    ref_loader = VideoLoader(reference_video_path, roi=roi)
    ref_info = ref_loader.info()

    main_loader = VideoLoader(main_video_path, skip_frames=skip_frames, roi=roi)
    main_info = main_loader.info()

    ref_total = ref_info["frame_count"]
    main_total = main_info["frame_count"] // skip_frames

    total_work = ref_total + main_total

    # -----------------------------
    # STEP 1: Build reference pose
    # -----------------------------
    pose_detector = PoseDetector(model_complexity=model_complexity)
    ref_landmarks_list = []

    for i, frame in enumerate(ref_loader):
        if stop_event and stop_event.is_set():
            pose_detector.close()
            ref_loader.release()
            return []

        lm = pose_detector.process(frame)
        if lm is not None:
            norm = normalize_landmarks(lm)
            ref_landmarks_list.append(norm)

        if progress_callback and total_work > 0:
            progress_callback(i / total_work)

    pose_detector.close()
    ref_loader.release()

    if not ref_landmarks_list:
        raise ValueError("No pose detected in reference video!")

    reference_pose = build_reference_pose(ref_landmarks_list)

    # -----------------------------
    # STEP 2: Search main video
    # -----------------------------
    loader = main_loader
    info = loader.info()

    pose_detector = PoseDetector(model_complexity=model_complexity)
    detector = EventDetector(
        info["fps"],
        threshold=similarity_threshold,
        duration_sec=duration_sec,
        cooldown_sec=cooldown_sec,
    )

    offset = ref_total  # shift progress after reference phase

    for i, frame in enumerate(loader):
        if stop_event and stop_event.is_set():
            pose_detector.close()
            loader.release()
            return []

        lm = pose_detector.process(frame)
        if lm is None:
            continue

        norm = normalize_landmarks(lm)
        sim = similarity_score(norm, reference_pose)
        detector.update(sim)

        if progress_callback and total_work > 0:
            progress_callback((offset + i) / total_work)

    pose_detector.close()
    loader.release()

    # -----------------------------
    # STEP 3: Format output
    # -----------------------------
    start_frames = detector.get_start_frames()
    start_timestamps = [(f * skip_frames) / info["fps"] for f in start_frames]

    formatted_times = [str(timedelta(seconds=round(t))) for t in start_timestamps]

    # ensure progress ends at 100%
    if progress_callback:
        progress_callback(1.0)

    return formatted_times