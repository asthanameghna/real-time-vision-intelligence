#!/usr/bin/env python3
"""Run YOLO11 on a sample video and write an annotated MP4."""

from pathlib import Path

import cv2

from app.core.detector import ObjectDetector

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "test.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "data" / "outputs" / "detection_output.mp4"


def draw_detections(frame, detections):
    """Draw bounding boxes and class labels on a BGR frame."""
    for det in detections:
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        caption = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 220, 0), -1)
        cv2.putText(
            frame,
            caption,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), fourcc, float(fps), (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for: {OUTPUT_VIDEO}")

    detector = ObjectDetector()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector.detect(frame)
            annotated = draw_detections(frame, dets)
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
