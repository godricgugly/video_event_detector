# import cv2


# def select_roi(video_path: str):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         raise ValueError(f"Could not open video: {video_path}")

#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         raise ValueError("Could not read frame for ROI selection")

#     roi = cv2.selectROI("Select ROI", frame, showCrosshair=True)
#     cv2.destroyAllWindows()

#     if roi == (0, 0, 0, 0):
#         return None

#     return roi  # (x, y, w, h)

import cv2


def select_roi(video_path: str, max_width=1280, max_height=720):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read frame for ROI selection")

    original_h, original_w = frame.shape[:2]

    # Compute scale factor to fit screen
    scale = min(max_width / original_w, max_height / original_h, 1.0)

    resized = frame
    if scale < 1.0:
        resized = cv2.resize(frame, (int(original_w * scale), int(original_h * scale)))

    roi = cv2.selectROI("Select ROI", resized, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        return None

    x, y, w, h = roi

    # Scale ROI back to original resolution
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    return (x, y, w, h)