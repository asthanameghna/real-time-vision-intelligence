# Technical FAQ

Deep-dive answers grounded in **this repository’s** implementation (`run_detection.py`, `app/core/*`, `configs/*`). Where the README mentions broader roadmap items (API, Streamlit, Docker), those are **not** part of the documented frame loop unless noted.

---

## Why ByteTrack instead of DeepSORT?

**Pragmatic reasons for this codebase:**

1. **Ultralytics ships ByteTrack as a first-class `track()` path** with a maintained `bytetrack.yaml` — one call site (`YOLO.track`) with `persist=True`, no separate feature extractor or ReID model wiring.

2. **Association philosophy**: ByteTrack’s use of **low-score** detections helps maintain boxes through short occlusions and score flicker without a learned appearance embedding. DeepSORT’s strength is **ReID-assisted** recovery across longer gaps when visual embeddings are reliable; that adds model weight, inference cost, and tuning surface.

3. **MVP complexity**: DeepSORT-class stacks often pair a detector with a **metric embedding** network (and calibration). For a modular demo pipeline centered on **geometry rules on centroids**, ByteTrack + YOLO is a common industry default with fewer moving parts.

**When DeepSORT (or BoT-SORT, StrongSORT, etc.) would be justified**: heavy crowding, long occlusions, similar-looking subjects, camera where bbox IoU association alone drifts.

---

## Why YAML configs?

1. **Non-code tuning**: model path, confidence, trajectory length, stationary threshold, and **pixel geometry** (zones/lines) change per deployment without Python edits.

2. **Diffability**: zones and thresholds version cleanly in Git; reviewers see **what changed** in a PR without parsing code.

3. **Loader ergonomics**: `PyYAML` + `yaml.safe_load` is standard in Python ML repos (`requirements.txt` pins `PyYAML`).

**Tradeoff**: no schema validation layer (e.g., pydantic) in-repo — malformed YAML fails at runtime with `ValueError` / `KeyError` from manual checks.

---

## Why modularize detector / tracker / motion / events?

- **Detector** (`ObjectDetector` / `Detection`): pure `predict()` — useful for still-image or “boxes only” pipelines.
- **Tracker** (`ByteTrackTracker` / `TrackedObject`): **IDs + boxes** over time.
- **Motion** (`MotionEstimator` / `TrackMotionState`): **state estimation** on top of tracks (trajectory, velocity).
- **Events** (`EventEngine`): **domain rules** (policies, zones) isolated from PyTorch.

This mirrors production separation: **model inference** vs **temporal fusion** vs **business logic** — even when deployed as one process (`run_detection.py`).

**Note**: `run_detection.py` currently uses **only** `ByteTrackTracker`, not `ObjectDetector`, because `YOLO.track()` already performs detection internally.

---

## Why centroid tracking (vs foot point, bottom-center, etc.)?

The centroid here is **exactly** the bbox center computed in `tracker.py`. Reasons:

- **Zero extra inference** (no pose / segmentation).
- **Single definition** for all classes (person, vehicle, etc.).
- **Sufficient** for coarse polygon and line rules when boxes are reasonable.

**Downside**: for people, the bbox center is often **torso-high**, not feet — “standing on a line” vs “crossing with torso” can disagree with human intuition. Production systems often use **bottom edge midpoint** for ground contact heuristics.

---

## Why are track IDs large or “random”?

IDs are allocated by the **tracker implementation inside Ultralytics/ByteTrack**, not by this application. They are typically **monotonic integers** for new tracks in a session but **not** guaranteed to be small or dense; after many births/deaths you may see **large integers** depending on internal counters.

They are **not** semantic (not “person #3 in the database”) and **not stable across runs** or across camera reboots. Re-identification across cameras is out of scope.

---

## Why JSONL for events?

1. **Append-friendly**: each line is an independent JSON object — safe for streaming writers and log tailers.

2. **Crash resilience**: partial files still parse line-by-line; unlike a single JSON array, you do not lose the whole file if the process dies mid-write.

3. **Downstream tooling**: trivial ingestion with Spark, BigQuery, `jq`, Polars (`requirements.txt` includes Polars though the frame loop does not use it).

**Current caveat**: `run_detection.py` opens the file with `"w"`, so **each run overwrites** — not append mode. For 24/7 logging you would open with append and likely rotate files.

---

## Why frame-derived timestamps instead of wall-clock time?

`run_detection.py` uses:

```python
time_sec = frame_idx / float(fps)
```

**Reasons:**

1. **Determinism**: Reprocessing the same MP4 yields the **same** `timestamp` sequence (modulo FPS metadata correctness).

2. **Alignment with annotations**: Events line up with **frame numbers** and exported video timecode for debugging (“at 12.3s in the clip”).

3. **Offline-first**: Many CV pipelines are developed on files where **wall clock** is meaningless.

**Gap**: live camera streams would usually stamp with **capture time** (and handle clock skew, NTP, etc.) in addition to or instead of frame index.

---

## What are the limitations of the current implementation?

| Area | Limitation |
|------|------------|
| **IDs** | No cross-session persistence; motion history wiped when track drops from frame. |
| **Geometry** | Pixel-space only; no camera calibration or world coordinates. |
| **Events** | No `zone_exit`; line crossing has no directional semantics in JSON; edge/colinearity cases simplistic. |
| **Detection/tracking** | Single-class-agnostic rules; no ROI crop to reduce compute; ByteTrack params not exposed in YAML. |
| **I/O** | Synchronous single-threaded loop; GPU underutilization vs decode/async possible. |
| **Config** | No schema validation; FPS fallback hardcoded if property missing. |
| **Testing** | No automated tests in tree for geometry or pipeline (as of this doc). |

---

## What production improvements could be added later?

- **Calibration**: homography or full camera model; metric speed and ground-plane zones.
- **Filtering**: EMA / Kalman on centroids; foot-point or keypoint-based anchors for people.
- **Richer events**: directional line crossing, dwell time, speed thresholds, object-class filters.
- **Robust I/O**: threaded decode, GPU zero-copy paths, RTSP/WebRTC sources.
- **Serving**: gRPC/REST or Kafka for events; the README’s API/streaming layers would sit here.
- **Observability**: per-stage latency histograms, dropped-frame counters, model version in each event.
- **Auth & PII**: redaction, retention policy, secure transport.

---

## What are the latency bottlenecks?

Ordered by typical impact in a **single-threaded** `read → track → write` loop:

1. **`YOLO.track()`** — neural net forward + tracker association (GPU-bound when CUDA is available; CPU-bound otherwise).

2. **Video decode** — `cv2.VideoCapture` can be CPU-heavy depending on codec and build flags.

3. **`VideoWriter` encode** — `mp4v` software encoding per frame.

4. **Python overhead** — comparatively small vs inference at reasonable resolutions, but non-zero (JSON serialization per event, OpenCV drawing).

**Not** dominant today: `EventEngine` geometry (OpenCV point-in-polygon and a few segment tests per object).

---

## How would this scale to multi-camera systems?

**Data plane**: one **logical pipeline instance per stream** (process or thread) with its own `ByteTrackTracker` / `MotionEstimator` / `EventEngine` — **never** share a tracker across cameras; IDs would collide semantically.

**Control plane**: a supervisor assigns streams, health-checks workers, and aggregates events (often into a time-series DB or message bus) keyed by **`camera_id`** (not present in current JSON schema — you would add it at write time).

**Model serving**: shared **read-only** weights on disk; optionally a **single GPU batching server** with multi-stream scheduling (adds queueing latency but improves throughput).

**Calibration**: each camera carries its own zone YAML or homography matrix.

**Clocks**: wall-time synchronization (PTP/NTP) if correlating events across cameras (e.g., multi-view re-ID or triangulation—out of current scope).
