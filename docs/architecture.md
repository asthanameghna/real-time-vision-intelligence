# System Architecture

This document describes the **implemented** perception stack in this repository: YOLO11 detection, ByteTrack association, centroid-based motion estimation, a rule-based event engine, YAML configuration, an OpenCV MP4 rendering path, and JSONL event logging. The entry point is `run_detection.py`.

---

## Overall Architecture

The system is a **single-process, synchronous frame loop**: each video frame is read once, processed through tracking → motion → events, annotated, and written to disk. There is no separate microservice layer, message bus, or GPU pipeline orchestration in the current code; the design favors clarity and a minimal integration surface for an MVP-style offline or batch demo.

```text
┌─────────────────┐
│ YAML configs    │──► paths, model, thresholds, motion params, zone geometry
└────────┬────────┘
         │
┌────────▼────────┐     ┌──────────────────┐
│ VideoCapture    │────►│ BGR frame (H×W×3) │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │              ┌────────▼─────────┐
         │              │ ByteTrackTracker │
         │              │ YOLO.track()     │
         │              └────────┬─────────┘
         │                       │ List[TrackedObject]
         │              ┌────────▼─────────┐
         │              │ MotionEstimator    │
         │              └────────┬─────────┘
         │                       │ Dict[id, TrackMotionState]
         │              ┌────────▼─────────┐
         │              │ EventEngine        │
         │              └────────┬─────────┘
         │                       │ List[event dict]
         │              ┌────────┴─────────┐
         │              ▼                  ▼
         │        JSONL append      OpenCV overlays
         │              │                  │
         └──────────────┴──────────────────┴──► VideoWriter (MP4)
```

---

## Module Responsibilities

| Module | Path | Responsibility |
|--------|------|----------------|
| **Orchestration** | `run_detection.py` | Load YAML, open video I/O, instantiate tracker/motion/event engine, advance `frame_idx`, compute `time_sec`, invoke pipeline per frame, flush JSONL lines, draw overlays, write frames. |
| **Tracking (detect + associate)** | `app/core/tracker.py` | Wraps Ultralytics `YOLO.track()` with `persist=True` and `tracker="bytetrack.yaml"`. Produces `TrackedObject` rows with `track_id`, `bbox`, `centroid`, `label`, `confidence`. |
| **Detection-only wrapper** | `app/core/detector.py` | `ObjectDetector` + `Detection` dataclass: `YOLO.predict()` without tracking. **Not used** by `run_detection.py` today; present for a detect-only code path or tests. |
| **Motion** | `app/core/motion.py` | Per-`track_id` deque of `(t, x, y)` centroid samples; finite-difference velocity between last two samples; speed in px/s; coarse direction enum (`left`/`right`/`up`/`down`/`stationary`). |
| **Events** | `app/core/events.py` | Parse zone/line YAML into `ZoneSpec` / `LineSpec`; `EventEngine.process_frame()` emits `zone_entry`, `line_crossing`, `occupancy_count`; drawing helpers for zones/lines and on-screen event ticker. |
| **Configuration** | `configs/default.yaml`, `configs/zones.yaml` | Model path, confidence, I/O paths, motion hyperparameters, polygon zones, line segments, optional occupancy zone subset. |

---

## Data Flow Between Components

1. **Frame → tracks**  
   A single `numpy` BGR array is passed to `ByteTrackTracker.track(frame)`. Ultralytics runs the configured YOLO11 weights, runs the ByteTrack post-processor, and returns boxes with optional `id` when `is_track` is true.

2. **Tracks → motion states**  
   `MotionEstimator.update(tracks, time_sec)` filters `track_id >= 0`, appends centroid samples keyed by ID, computes velocity from the last two samples, and **deletes** histories for IDs absent from the current frame’s track list (no long-term re-identification buffer).

3. **Tracks + motion → events**  
   `EventEngine.process_frame(tracks, motion_by_id, time_sec)` uses **current** centroids for polygon tests and **previous vs current** centroids (from `trajectory_xy`) for line-segment intersection. Emits zero or more event dicts per frame.

4. **Events → JSONL**  
   Each event is `json.dumps(ev) + "\n"` appended to the configured file in the same frame loop (line-buffered by the `with` block; no explicit flush per line).

5. **Visualization**  
   The same frame is mutated in place: zones/lines, polylines for trajectories, boxes, labels, then a small HUD of recent events.

---

## Perception Pipeline Explanation

The **perception pipeline** in the strict sense is: **appearance-based detector** → **motion-model-free multi-object tracker** → **geometric rules on 2D points**.

- **Detector**: YOLO11 produces class-conditioned boxes and scores. The tracker consumes these every frame; there is no separate Kalman filter implementation in this repo (ByteTrack’s internal Kalman lives inside Ultralytics’ integration).

- **Tracker**: ByteTrack associates high-confidence and low-confidence detections to maintain tracks through short occlusions and inconsistent detections better than naive greedy IoU matching. The project does not expose ByteTrack hyperparameters in YAML; they come from Ultralytics’ `bytetrack.yaml`.

- **Motion layer**: A deliberately **lightweight** second stage: no learned motion model, only sampled centroids and discrete-time differentiation.

- **Event layer**: Pure **symbolic geometry** (point-in-polygon, segment intersection) plus a small amount of **finite-state memory** (last inside/outside per `(track, zone)`, pending line state, last occupancy count).

---

## Why Modular Design Was Used

1. **Replaceability**: `ByteTrackTracker` could be swapped for another backend that still yields `TrackedObject`-shaped outputs without touching motion or events.

2. **Testability**: Zone parsing (`load_zone_specs`), segment intersection, and motion differentiation can be reasoned about independently of GPU inference.

3. **Cognitive load**: `run_detection.py` stays a thin composition root; domain logic lives in small modules aligned with pipeline stages.

4. **Recruiter / reviewer signal**: The boundaries mirror how production systems separate **model inference**, **state estimation**, and **business rules**—even though deployment here is monolithic.

---

## Temporal State

Temporal state is any memory carried across frames. In this codebase it lives in three places:

| Location | State | Purpose |
|----------|--------|---------|
| **Ultralytics tracker** (`persist=True`) | Internal track table, Kalman states, lost-track buffers | Stable `track_id` assignment across frames (implementation detail inside the library). |
| **`MotionEstimator._histories`** | `Dict[track_id, Deque[_Sample]]` | Trajectory and velocity; **cleared** when a track ID disappears from the current frame. |
| **`EventEngine`** | `_was_inside_zone`, `_line_cross_pending`, `_last_occupancy` | Edge-triggered zone entry, one-shot line crossing until the centroid path uncrosses, occupancy only on count change. |

**Important distinction**: video **timeline** uses `time_sec = frame_idx / fps` from `run_detection.py`, not `time.time()`. That ties semantics to the **media timeline** (deterministic replays, A/B comparisons) rather than wall clock.

There is **no** global world model, no map from pixel space to meters, and no cross-camera identity—only per-stream IDs and pixel geometry.
