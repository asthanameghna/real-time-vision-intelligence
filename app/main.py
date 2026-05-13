import json
import math
import time
from pathlib import Path

import cv2
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

app = FastAPI()

PROCESSED_FRAMES = 0
ACTIVE_TRACKS = 0

_ROOT = Path(__file__).resolve().parent.parent
_EVENTS_PATH = _ROOT / "data" / "outputs" / "events.jsonl"
_DEFAULT_CONFIG_PATH = _ROOT / "configs" / "default.yaml"


def _count_valid_event_lines() -> int:
    if not _EVENTS_PATH.is_file():
        return 0
    count = 0
    for line in _EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            count += 1
        except json.JSONDecodeError:
            continue
    return count


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def events():
    if not _EVENTS_PATH.is_file():
        return []

    lines = _EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    parsed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return parsed[-10:]


@app.get("/metrics")
def metrics():
    return {
        "service_name": "real-time-vision-intelligence",
        "processed_frames": PROCESSED_FRAMES,
        "fps": 0.0,
        "active_tracks": ACTIVE_TRACKS,
        "total_events": _count_valid_event_lines(),
    }


@app.get("/frame")
def frame():
    if not _DEFAULT_CONFIG_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="config not found: configs/default.yaml",
        )
    with _DEFAULT_CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("input_video")
    if not raw or not isinstance(raw, str):
        raise HTTPException(
            status_code=500,
            detail="configs/default.yaml missing valid input_video string",
        )
    video_path = Path(raw) if Path(raw).is_absolute() else _ROOT / raw
    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"input video file not found: {video_path}",
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=503,
            detail=f"could not open video for reading: {video_path}",
        )
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise HTTPException(
            status_code=503,
            detail=f"could not read first frame from: {video_path}",
        )
    ret, buf = cv2.imencode(".jpg", img)
    if not ret or buf is None:
        raise HTTPException(
            status_code=500,
            detail="could not encode frame as JPEG",
        )
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def _mjpeg_frame_chunks(video_path: Path, config_path: Path):
    global PROCESSED_FRAMES, ACTIVE_TRACKS

    from collections import deque

    from app.core.events import (
        draw_recent_event_alerts,
        draw_zones_and_lines,
        load_zone_specs,
    )
    from app.core.pipeline import VisionPipeline
    from run_detection import draw_tracks, load_config, resolve_path

    cfg = load_config(config_path)
    model = str(cfg["model"])
    conf = float(cfg["confidence_threshold"])
    motion_cfg = cfg.get("motion") or {}
    if not isinstance(motion_cfg, dict):
        motion_cfg = {}
    max_traj = int(motion_cfg.get("max_trajectory_points", 64))
    stationary_pps = float(motion_cfg.get("stationary_speed_pps", 25.0))

    zones_path = resolve_path(str(cfg.get("zones_config", "configs/zones.yaml")))
    with zones_path.open(encoding="utf-8") as zf:
        zones_data = yaml.safe_load(zf)
    if not isinstance(zones_data, dict):
        return
    zone_specs, line_specs, occ_zone_ids = load_zone_specs(zones_data)

    pipeline = VisionPipeline(
        model_path=model,
        conf_threshold=conf,
        max_trajectory_points=max_traj,
        stationary_speed_pps=stationary_pps,
        zone_specs=zone_specs,
        line_specs=line_specs,
        occupancy_zone_ids=occ_zone_ids,
    )
    recent_events: deque = deque(maxlen=12)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    frame_idx = 0
    try:
        while True:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
            delay = 1.0 / fps

            ok, img = cap.read()
            if not ok or img is None:
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    break
                frame_idx = 0
                pipeline.reset_state()
                recent_events.clear()
                continue

            time_sec = frame_idx / fps
            tracks = pipeline.tracker.track(img)
            ACTIVE_TRACKS = len(tracks)
            motion_by_id = pipeline.motion.update(tracks, time_sec)
            for ev in pipeline.event_engine.process_frame(
                tracks, motion_by_id, time_sec
            ):
                recent_events.append(ev)

            draw_zones_and_lines(img, zone_specs, line_specs)
            draw_tracks(img, tracks, motion_by_id)
            draw_recent_event_alerts(img, recent_events)
            frame_idx += 1

            ret, buf = cv2.imencode(".jpg", img)
            if not ret or buf is None:
                continue
            PROCESSED_FRAMES += 1
            yield boundary + buf.tobytes() + b"\r\n"
            time.sleep(delay)
    finally:
        cap.release()


@app.get("/stream")
def stream():
    from app.core.events import load_zone_specs
    from run_detection import load_config, resolve_path

    if not _DEFAULT_CONFIG_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="config not found: configs/default.yaml",
        )
    try:
        cfg = load_config(_DEFAULT_CONFIG_PATH)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    raw = cfg.get("input_video")
    if not raw or not isinstance(raw, str):
        raise HTTPException(
            status_code=500,
            detail="configs/default.yaml missing valid input_video string",
        )
    video_path = Path(raw) if Path(raw).is_absolute() else _ROOT / raw
    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"input video file not found: {video_path}",
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=503,
            detail=f"could not open video for reading: {video_path}",
        )
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=f"could not read first frame from: {video_path}",
        )

    try:
        str(cfg["model"])
        float(cfg["confidence_threshold"])
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"configs/default.yaml missing required key: {e.args[0]}",
        ) from e

    zones_path = resolve_path(str(cfg.get("zones_config", "configs/zones.yaml")))
    if not zones_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"zones config not found: {zones_path}",
        )
    try:
        with zones_path.open(encoding="utf-8") as zf:
            zd = yaml.safe_load(zf)
        if not isinstance(zd, dict):
            raise ValueError("zones config must be a mapping")
        load_zone_specs(zd)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return StreamingResponse(
        _mjpeg_frame_chunks(video_path, _DEFAULT_CONFIG_PATH),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
