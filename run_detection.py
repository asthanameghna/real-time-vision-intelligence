#!/usr/bin/env python3
"""Run YOLO11 + ByteTrack on a sample video and write an annotated MP4."""

from pathlib import Path

import cv2
import numpy as np
import yaml

from app.core.motion import MotionEstimator, TrackMotionState
from app.core.tracker import ByteTrackTracker, TrackedObject

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def draw_tracks(
    frame,
    tracks: list[TrackedObject],
    motion_by_id: dict[int, TrackMotionState],
) -> None:
    """Draw boxes, trajectories, labels, and per-track velocity on a BGR frame (in place)."""
    for t in tracks:
        x1, y1, x2, y2 = int(t.bbox[0]), int(t.bbox[1]), int(t.bbox[2]), int(t.bbox[3])
        m = motion_by_id.get(t.track_id)

        if m is not None and len(m.trajectory_xy) >= 2:
            pts = np.array(
                [[[int(px), int(py)] for px, py in m.trajectory_xy]],
                dtype=np.int32,
            )
            cv2.polylines(
                frame, pts, isClosed=False, color=(255, 165, 0), thickness=2, lineType=cv2.LINE_AA
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

        if m is not None and t.track_id >= 0:
            caption = (
                f"{t.label} {t.confidence:.2f} ID:{t.track_id} "
                f"{m.speed_pps:.0f}px/s {m.direction.value}"
            )
        else:
            caption = f"{t.label} {t.confidence:.2f} ID:{t.track_id}"

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


def main() -> None:
    if not DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {DEFAULT_CONFIG_PATH}")

    cfg = load_config(DEFAULT_CONFIG_PATH)
    try:
        model = str(cfg["model"])
        conf = float(cfg["confidence_threshold"])
        input_video = resolve_path(str(cfg["input_video"]))
        output_video = resolve_path(str(cfg["output_video"]))
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e.args[0]}") from e

    motion_cfg = cfg.get("motion") or {}
    if not isinstance(motion_cfg, dict):
        raise ValueError("Config key 'motion' must be a mapping when present")
    max_traj = int(motion_cfg.get("max_trajectory_points", 64))
    stationary_pps = float(motion_cfg.get("stationary_speed_pps", 25.0))

    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video), fourcc, float(fps), (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for: {output_video}")

    tracker = ByteTrackTracker(model_path=model, conf_threshold=conf)
    motion = MotionEstimator(
        max_trajectory_points=max_traj,
        stationary_speed_pps=stationary_pps,
    )

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_sec = frame_idx / float(fps)
            tracks = tracker.track(frame)
            motion_by_id = motion.update(tracks, time_sec)
            draw_tracks(frame, tracks, motion_by_id)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {output_video}")


if __name__ == "__main__":
    main()
