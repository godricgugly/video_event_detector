from src.app.pipeline import run_detection
from src.video.roi import select_roi

if __name__ == "__main__":
    reference_video_path = "data/references/ref_pose.mp4"
    main_video_path = "data/raw/session.mp4"

    roi = select_roi(main_video_path)

    results = run_detection(
        reference_video_path,
        main_video_path,
        roi=roi,
    )

    print("Event starts at:", results)