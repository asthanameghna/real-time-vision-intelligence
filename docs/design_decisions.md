# Design Decisions

This document records **major engineering choices** visible in the implementation, the **tradeoffs** accepted for MVP simplicity, **technology rationale**, and plausible **evolution paths**. It is written for engineers and technical interviewers who want the “why,” not a feature brochure.

---

## 1. Ultralytics as the single integration point for YOLO + ByteTrack

**Decision**: Use `ultralytics.YOLO.track()` with `tracker="bytetrack.yaml"` and `persist=True` instead of maintaining a separate detector executable and a standalone ByteTrack repo fork.

**Why**

- **Velocity**: one dependency (`ultralytics` pulls `torch`, uses well-tested `track` API).
- **Weight management**: automatic model discovery/download patterns users expect from YOLO tooling.
- **Maintenance**: tracker YAML and version coupling stay aligned with the Ultralytics release pinned in `requirements.txt`.

**Tradeoff**

- **Opacity**: Kalman parameters, track birth/death, and low-score association live **inside** the library — harder to audit than a local fork.
- **Version lock-in**: upgrading `ultralytics` can change tracking behavior without code diffs in this repo.

**Evolution**

- Swap to **native TensorRT/ONNX** runtime with a **thin** tracker adapter if latency requirements tighten.
- Expose **tracker YAML path** in `default.yaml` for ops tuning without code changes.

---

## 2. Separate `ObjectDetector` vs `ByteTrackTracker`

**Decision**: Keep `app/core/detector.py` (`YOLO.predict`) separate from `app/core/tracker.py` (`YOLO.track`), but wire **only** the tracker in `run_detection.py`.

**Why**

- **Conceptual clarity**: “boxes without IDs” vs “boxes with IDs” are different products.
- **Future reuse**: detection-only benchmarks, two-stage pipelines (detect → custom associate), or unit tests without tracker state.

**Tradeoff**

- **Redundant model load** if both were ever used in one process without refactoring to share weights.

**Evolution**

- Factory that returns either pipeline based on config, sharing one `YOLO` instance where possible.

---

## 3. Centroid motion on bbox centers

**Decision**: `MotionEstimator` samples `(cx, cy)` from bbox midpoints; velocity is two-point finite difference; direction is axis-dominant with a stationary band.

**Why**

- **Minimal assumptions**: works for arbitrary COCO-like classes without per-class keypoint models.
- **Predictable cost**: O(1) per track per frame.

**Tradeoff**

- **Semantic mismatch** for “ground” analytics; jittery boxes inject noise into speed.

**Evolution**

- Class-conditional anchor points (e.g., bottom-center for `person`).
- Optional **Savitzky–Golay** or **EMA** smoothing without changing the event API.

---

## 4. Rule-only event engine (no ML for “behavior”)

**Decision**: `EventEngine` uses polygons, line segments, counts, and explicit finite state — no learned activity recognizer.

**Why**

- **Explainability**: stakeholders can map events to drawn regions on a still frame.
- **Debuggability**: failures are geometric or threshold bugs, not opaque model errors.
- **Dataset independence**: no need for behavior labels to ship a demo.

**Tradeoff**

- **Brittleness** to camera angle, zone drawing error, and ambiguous interactions (crowding, partial occlusion).

**Evolution**

- Hybrid: rules for **hard constraints** (ROI gating) + small classifier on **track snippets** for complex behaviors.

---

## 5. YAML for geometry and hyperparameters

**Decision**: `configs/default.yaml` + `configs/zones.yaml`, parsed with PyYAML, validated only by imperative checks.

**Why**

- **Fast iteration** for non-developers (ops, field engineers) adjusting ROI polygons.
- **Git-friendly** diffs.

**Tradeoff**

- **No schema**: typos surface late; optional keys have subtle defaults (e.g., occupancy defaults to all zones when `occupancy` omitted).

**Evolution**

- Pydantic / JSON Schema validation at startup with actionable error messages.

---

## 6. JSONL events with UUID primary keys

**Decision**: Each event gets `uuid.uuid4()`; logs are newline-delimited JSON.

**Why**

- **Stream processing** and **partial reads** are straightforward.
- **UUID** avoids client-side ID coordination in a future distributed system.

**Tradeoff**

- **Non-reproducible IDs** across runs complicate golden-file testing — you must strip `event_id` for diffs.
- Opening output with `"w"` **truncates** each run — good for demos, bad for append-only production logging.

**Evolution**

- Append mode + date-partitioned files + optional **deterministic** IDs `(camera_id, frame_idx, track_id, type, zone_id)`.

---

## 7. Timeline from `frame_idx / fps`

**Decision**: Event `timestamp` uses synthetic timeline from frame index and container FPS, not `time.time()`.

**Why**

- **Reproducible** science/engineering workflows on files.
- Matches how **video players** seek.

**Tradeoff**

- Wrong **FPS metadata** poisons velocity and timestamps silently (except the hardcoded fallback when FPS is falsy).

**Evolution**

- Per-frame **PTS** from demuxer when available; wall-clock **ingest timestamp** as separate fields.

---

## 8. OpenCV-centric I/O and visualization

**Decision**: `VideoCapture` / `VideoWriter`, in-place drawing, `mp4v` FourCC.

**Why**

- **Ubiquity**: lowest friction for a portable demo.
- **No extra UI framework** required for the core pipeline.

**Tradeoff**

- **Decode/encode performance** and **codec flexibility** lag behind ffmpeg-first or GPU pipelines.

**Evolution**

- Decouple **decode thread**, **inference thread**, and **encode thread** with bounded queues (backpressure-aware).

---

## 9. MVP scope vs README roadmap

**Decision (implicit in repo layout)**: The runnable path is `run_detection.py` + `app/core/*` + YAML + JSONL + MP4 out. README lists FastAPI, WebSockets, Streamlit, Docker — **aspirational / planned**, not wired in the same module graph as the frame loop documented here.

**Why** (typical pattern)

- Establish **correctness** on offline video before operationalizing streaming.

**Risk**

- Readers confuse **documented** vs **planned** capabilities — these `docs/` files anchor on **code that exists**.

---

## Summary Table

| Area | MVP choice | Production evolution |
|------|------------|----------------------|
| Tracking | Ultralytics ByteTrack | Tuned YAML, alternate trackers, batch inference |
| Motion | Raw centroid diff | Smoothing, foot point, metric calibration |
| Events | Rules + minimal state | Richer semantics, schema validation, dedupe policies |
| Time | Frame index / FPS | PTS + wall clock, sync across cameras |
| I/O | OpenCV sync loop | Async decode/encode, RTSP, hardware encoders |
