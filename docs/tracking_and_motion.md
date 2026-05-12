# Tracking and Motion Estimation

This document explains how **ByteTrack** is integrated in this project, why `persist=True` matters, how centroids and trajectories are formed, how velocity and direction are derived, and the limitations of **pixel-space** motion.

Implementation references: `app/core/tracker.py`, `app/core/motion.py`.

---

## ByteTrack: Conceptual Overview

ByteTrack (Zhang et al.) improves multi-object tracking in cluttered video by **using both high- and low-score detections** in association. High-confidence boxes anchor identities; low-confidence boxes help **bridge gaps** when objects are partially occluded, motion-blurred, or briefly missed at the default score threshold—cases where a tracker that discards low-score boxes would drop tracks.

The Ultralytics integration bundles:

- A **Kalman filter** on box state (implementation inside the library, not in this repo).
- **Hungarian / LAP** assignment (project depends on `lap` via `requirements.txt`).
- Track birth/death and “lost” track policies as defined in `bytetrack.yaml` shipped with Ultralytics.

This project does **not** fork or reimplement ByteTrack; it configures `YOLO.track(..., tracker="bytetrack.yaml")`.

---

## Why `persist=True` Matters

In `ByteTrackTracker.track()`:

```python
results = self._model.track(
    source=frame,
    conf=self._conf_threshold,
    persist=True,
    tracker=TRACKER_CFG,
    verbose=False,
)
```

**`persist=True`** tells Ultralytics to **reuse the same tracker state** across successive calls on this model instance. With `persist=False`, tracker internal state would reset each frame—IDs would not be stable, velocity from consecutive positions would be meaningless, and the event engine’s per-`track_id` memory would churn constantly.

**Operational rule**: construct **one** `ByteTrackTracker` (one `YOLO` instance) and call `track(frame)` **strictly in temporal order** for a given video stream.

---

## Centroid Extraction

For each tracked box with corners `(x1, y1, x2, y2)`:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

**Why bbox center?**

- **Cheap and deterministic** — no instance segmentation or pose keypoints.
- **Aligned with many analytics definitions** (e.g., “person center” heuristics).
- **Stable enough** when boxes are tight, problematic when boxes are wide (person stretching arms) or when the feet should define “standing in zone” semantics.

**Coordinate convention**: OpenCV / image space — **origin top-left**, **x** rightward, **y** downward. This propagates to velocity and direction labels.

---

## Trajectory History

`MotionEstimator` maintains:

```python
self._histories: Dict[int, Deque[_Sample]]  # _Sample: t, x, y
```

Each update appends one sample per active track. The deque has `maxlen=max_trajectory_points` (default **64** from `configs/default.yaml`), so old points roll off automatically.

**Stale cleanup**: IDs not present in the current `tracks` list have their deque **removed**:

```python
stale = [tid for tid in self._histories if tid not in active_ids]
for tid in stale:
    del self._histories[tid]
```

Implications:

- If a person **leaves the frame** and returns later, they may get a **new** `track_id` — trajectory does not span absence.
- There is **no** trajectory retained for `track_id == -1` rows (they are skipped entirely).

---

## Velocity Estimation

Given the last two samples `s0`, `s1` in the deque:

```text
dt = s1.t - s0.t
vx = (s1.x - s0.x) / dt
vy = (s1.y - s0.y) / dt
speed = sqrt(vx² + vy²)
```

With `time_sec = frame_idx / fps`, for steady frame intervals `dt ≈ 1/fps`, so velocity is a **per-frame displacement scaled by FPS** — exactly **pixels per second** in the idealized constant-FPS model.

**Noise sensitivity**: a single jittery box causes a spike in `speed_pps` and can flip `direction` if it crosses the stationary threshold boundary.

---

## Direction Classification

`_direction_from_velocity` applies:

1. If `speed < stationary_speed_pps` → `stationary` (default threshold in YAML: **25.0** px/s; note `MotionEstimator`’s Python default is 12.0 if instantiated without config — the **run script** passes YAML values).

2. Else compare `|vx|` vs `|vy|`:
   - Larger horizontal component → `left` if `vx < 0` else `right`
   - Larger vertical component → `up` if `vy < 0` else `down` (remember **+y is down**, so `vy < 0` means moving toward the **top** of the image)

This is a **cardinal bucket**, not a heading angle; diagonal motion is classified by whichever axis dominates.

---

## Limitations of Pixel-Space Velocity

| Limitation | Consequence |
|------------|-------------|
| **Not metric** | `px/s` depends on camera resolution, field of view, and distance to subject. Two identical walking speeds can yield different `speed_pps`. |
| **No homography** | Ground plane motion is not rectified; oblique views distort speed along the image plane. |
| **BBox vs true motion** | Pose changes, arm swing, or detector box breathing add high-frequency noise to centroids. |
| **FPS uncertainty** | `CAP_PROP_FPS` can be wrong or zero for some files; `run_detection.py` falls back to `25.0`, mis-scaling time and all derivatives. |
| **Single-step derivative** | No low-pass filter; velocity is maximally local. |

For production traffic or safety analytics, you would typically calibrate **homography** or **camera extrinsics**, filter velocity (e.g., EMA, Kalman on centroid), and/or estimate **3D foot points** from a person model.
