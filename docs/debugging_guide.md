# Debugging Guide

This guide targets engineers running **`python run_detection.py`** and iterating on `configs/default.yaml`, `configs/zones.yaml`, and `app/core/*`.

---

## Common Failure Points

### 1. Config or path errors

| Symptom | Likely cause | Where it fails |
|---------|----------------|----------------|
| `FileNotFoundError: Config not found` | Missing `configs/default.yaml` | `run_detection.py` startup |
| `KeyError: Missing required config key` | Typo or omission (`model`, `confidence_threshold`, `input_video`, `output_video`) | `main()` |
| `FileNotFoundError: Input video not found` | Wrong `input_video` relative path | Before `VideoCapture` |
| `FileNotFoundError: Zones config not found` | `zones_config` path wrong | Before zone load |
| `ValueError: Config key 'motion' must be a mapping` | `motion:` set to a scalar (e.g., string) | `main()` |

Paths are resolved with `PROJECT_ROOT / relative` unless absolute (`resolve_path`).

### 2. Zone / YAML content errors

| Symptom | Likely cause |
|---------|----------------|
| `ValueError: 'zones' must be a list` | `zones:` is a mapping instead of a list |
| `ValueError: polygon must have at least 3 vertices` | Degenerate polygon |
| `ValueError: Expected [x, y]` | Malformed point (e.g., 3 numbers, string) |

### 3. Video writer failure

`RuntimeError: Could not open writer for: ...` — `VideoWriter.isOpened()` false. Typical causes:

- Output directory could not be created (permissions) — though `mkdir(parents=True)` usually succeeds.
- Invalid path or unsupported combination of FourCC + extension on the platform.
- Codec not available in the OpenCV build.

---

## Dependency Issues

Pinned stack is in `requirements.txt` (notably `torch==2.11.0`, `ultralytics==8.4.48`, `opencv-python==4.13.0.92`, `numpy==2.4.4`).

**Symptoms:**

- **ImportError for `cv2`**: OpenCV not installed in the active environment.
- **`ultralytics` model download hangs or fails**: network blocked; first run needs to fetch `yolo11n.pt` if not present.
- **CUDA mismatch**: PyTorch wheel installed for wrong CUDA runtime — inference falls back to CPU or errors at load; check `torch.cuda.is_available()` in a REPL.
- **`lap` build failures** on exotic platforms: ByteTrack association depends on `lap`; use a supported platform wheel or build toolchain.

**Isolation tip**: use a fresh venv and `pip install -r requirements.txt` exactly; mixing conda/pip for `torch` + `opencv` often causes symbol or ABI issues.

---

## Tracker Instability

**Symptoms**: IDs swap between objects, flicker `-1` IDs, fragmented tracks.

**Checklist:**

1. **`persist=True`** must remain enabled in `ByteTrackTracker.track()` — if you accidentally call `track()` on a **new** `YOLO` instance each frame, IDs will not persist.

2. **Confidence threshold**: Lower `confidence_threshold` in YAML → more weak boxes → more association ambiguity **or** better gap-filling — tune per scene.

3. **Resolution and motion blur**: fast motion + low FPS → IoU drops; consider higher FPS input or model size (`yolo11n` vs larger variants).

4. **Class confusion**: similar overlapping classes confuse the detector first; the tracker inherits those errors.

5. **`track_id == -1`**: Ultralytics did not assign an ID (`is_track` false). Downstream code **skips** these for motion/events — if most boxes are `-1`, investigate tracker config or model outputs.

**Evidence gathering**: dump per-frame `len(tracks)` and ID sets to a CSV for a short clip; visualize raw `track()` boxes without motion smoothing to separate detector noise from association noise.

---

## Path / Config Errors (Zones vs Video Resolution)

Zones in `configs/zones.yaml` are **pixel coordinates** in the **decoded frame** space. If coordinates were authored for 1920×1080 but the video is 1280×720, **no automatic scaling** occurs — zones float in the wrong place or off-frame.

**Fix**: author zones using a frame grab from the **actual** input (`cv2.imwrite` a snapshot) or implement scaling (not present today).

---

## Video Codec Issues

**Reading:**

- Some H.264/H.265 files fail depending on OpenCV’s FFmpeg backend — symptom: `cap.read()` always false or green frames.
- **Mitigation**: re-encode with ffmpeg (`libx264`, yuv420p) to a conservative profile; or use a different capture backend.

**Writing:**

- Current code uses `cv2.VideoWriter_fourcc(*"mp4v")` — widely compatible but **large files** and not always browser-friendly.
- If the player shows black: try `avc1` / H.264 on macOS with appropriate OpenCV build, or post-process with ffmpeg.

---

## OpenCV Rendering Issues

| Symptom | Cause / fix |
|---------|-------------|
| No trajectories | Need `len(m.trajectory_xy) >= 2` and valid `track_id`; new tracks need two frames. |
| No event HUD | `recent_events` empty — no events fired; check zones/lines and thresholds. |
| Wrong colors | BGR order — colors are `(B,G,R)` tuples in code. |
| Text garbled | Font scale / thickness; very small frames clip the HUD (`draw_recent_event_alerts` caps box width). |

Drawing mutates `frame` **in place** — if you refactor to parallel paths, copy the array before drawing.

---

## Debugging Strategies for Perception Systems

1. **Freeze time**: process **N frames** (e.g., 30–120) and save **raw** vs **annotated** side-by-side — narrows whether bugs are in decode, model, or rules.

2. **Single-object clip**: one person walking a line — validates line crossing and pending state without multi-object confusion.

3. **Geometry overlay first**: run with `draw_zones_and_lines` mentally verified against a still frame before trusting events.

4. **Event–video alignment**: `timestamp` equals `frame_idx / fps`; locate frame `round(timestamp * fps)`.

5. **Binary search thresholds**: adjust `confidence_threshold` and `stationary_speed_pps` independently to see which layer misbehaves.

6. **Minimal repro**: smallest MP4 + smallest YAML that reproduces the bug — attach to an issue with `ffprobe` output for the video stream.

7. **Profiler**: if latency is unclear, wrap `tracker.track`, `motion.update`, `event_engine.process_frame`, and `writer.write` with `time.perf_counter()` logs (not in repo by default).

---

## JSONL / Logging Pitfalls

- **Overwrite**: each run opens `events_output` with `"w"` — you may think “no events” when the file was truncated at the start of a failed run.
- **Partial lines**: if the process is killed mid-`write`, the last line may be incomplete — stream parsers should tolerate bad lines or validate JSON per line.

---

## When to Instrument vs When to Replace

- **Instrument** when metrics are needed (per-stage ms, track count, event rate).
- **Replace** when fundamental limits hit (e.g., need directional line logic, 3D world positions, or multi-camera correlation) — see `docs/design_decisions.md` and `docs/technical_faq.md`.
