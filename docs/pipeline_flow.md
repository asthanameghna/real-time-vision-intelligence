# Pipeline Flow: Frame Lifecycle

This document walks through **exactly** what happens for each frame when you run `python run_detection.py`, using the current implementation.

---

## Startup (Once)

1. **Resolve `configs/default.yaml`** relative to project root (`PROJECT_ROOT`). Required keys: `model`, `confidence_threshold`, `input_video`, `output_video`. Optional: `zones_config` (default `configs/zones.yaml`), `events_output` (default `data/outputs/events.jsonl`), `motion` block.

2. **Load zones** from YAML via `yaml.safe_load`, then `load_zone_specs()` → `zone_specs`, `line_specs`, `occ_zone_ids` (or inferred defaults for occupancy).

3. **Construct** `EventEngine(zone_specs, line_specs, occupancy_zone_ids=occ_zone_ids)`.

4. **OpenCV I/O**: `VideoCapture` on input; read width, height, FPS (`CAP_PROP_FPS` with fallback `25.0`); `VideoWriter` with FourCC `mp4v`.

5. **Construct** `ByteTrackTracker(model_path=model, conf_threshold=conf)`, `MotionEstimator(...)`, and `recent_events: deque(maxlen=12)` for the HUD.

6. **Open** the events file in write mode (`"w"`) — **each run truncates** the JSONL file.

---

## Per-Frame Loop

For each iteration until `cap.read()` fails:

### Step 1 — Read frame

```python
ok, frame = cap.read()
```

`frame` is BGR, `uint8`, shape `(H, W, 3)`.

### Step 2 — Compute timeline timestamp

```python
time_sec = frame_idx / float(fps)
```

**Not** wall-clock time. The first frame uses `frame_idx == 0` → `t = 0` (assuming FPS is trusted).

### Step 3 — Tracking (`ByteTrackTracker.track`)

- Calls `self._model.track(...)` with `persist=True`, `tracker="bytetrack.yaml"`, `conf=self._conf_threshold`.
- Returns `list[TrackedObject]`. Each item includes:
  - `bbox` `(x1,y1,x2,y2)` in pixel floats
  - `centroid` `((x1+x2)/2, (y1+y2)/2)`
  - `label`, `confidence`, `class`-derived label
  - `track_id`: integer from box `id` if `d.is_track`, else **`-1`**

Tracks with `track_id == -1` are **ignored** by motion and by most event logic (centroid events require valid IDs).

### Step 4 — Motion (`MotionEstimator.update`)

Input: current `tracks`, `time_sec`.

For each track with `track_id >= 0`:

1. Ensure a `deque(maxlen=max_trajectory_points)` exists for that ID.
2. Append `_Sample(t=time_sec, x=cx, y=cy)`.
3. If `len(hist) >= 2`, set `vx, vy` from the **last two** samples:  
   `(s1 - s0) / (s1.t - s0.t)` when `dt > 1e-9`.
4. `speed = hypot(vx, vy)` in **pixels per second** (because `time_sec` is in seconds).
5. `direction` from dominant axis vs `stationary_speed_pps` threshold.

**Prune**: any `track_id` in `_histories` that is **not** in the current frame’s active ID set is **deleted** entirely — no ghost trajectories for lost tracks.

Output: `motion_by_id: Dict[int, TrackMotionState]`.

### Step 5 — Events (`EventEngine.process_frame`)

Input: `tracks`, `motion_by_id`, `time_sec`.

Per track (with valid ID):

1. **Zone entry (edge-triggered)**  
   For each `ZoneSpec`, `inside = point_in_polygon(centroid, polygon)`.  
   If `inside and not was_inside` → emit `zone_entry`, then set `was_inside` true.

2. **Line crossing**  
   Requires `len(trajectory_xy) >= 2`. Previous and current centroids form a segment; test intersection with line segment `(p1, p2)`.  
   If intersecting and `(track_id, line_id)` not in `_line_cross_pending` → emit `line_crossing` and add pending.  
   If **not** intersecting → remove pending for that pair (allows a future crossing to fire again).  
   Special case: pending entries for tracks that disappeared this frame are discarded.

3. **Occupancy (level-triggered with change detection)**  
   For each zone ID in the occupancy set, count tracks whose centroid is inside the polygon.  
   If count differs from `_last_occupancy[zone_id]` (or first observation) → emit `occupancy_count` and update cache.

Output: `list[dict]` (possibly empty).

### Step 6 — Persist events

For each event dict: `events_f.write(json.dumps(ev) + "\n")` and `recent_events.append(ev)`.

### Step 7 — Render (in-place on `frame`)

1. `draw_zones_and_lines` — closed polylines for zones, thick lines for line specs.
2. `draw_tracks` — trajectory polyline (if ≥ 2 points), rectangle, label with speed/direction when motion exists.
3. `draw_recent_event_alerts` — semi-transparent panel with last few events.

### Step 8 — Write video

`writer.write(frame)` then `frame_idx += 1`.

---

## How Detections Become Tracks

Inside Ultralytics, each frame’s YOLO **detections** are passed to the ByteTrack **association** stage, which outputs boxes annotated with persistent IDs when tracking succeeds.

At the Python boundary, the distinction between “raw detection” and “track” is exposed as `d.is_track`: if false, the wrapper assigns `track_id = -1`. The application treats `-1` as **non-tracked** noise for downstream modules.

---

## How Tracks Become Motion States

1. **Centroid extraction** (in `tracker.py`): geometric center of the axis-aligned bounding box — not a mask centroid or foot point.

2. **Sampling**: one `(t, x, y)` per frame per ID.

3. **Velocity**: backward Euler over the last interval — reactive to frame-to-frame jitter; no smoothing kernel beyond the implicit cap of `max_trajectory_points` ( deque stores history but velocity uses only the last step).

---

## How Motion States Become Events

| Event type | Primary signal | Secondary / memory |
|------------|----------------|---------------------|
| `zone_entry` | Current centroid inside polygon | `_was_inside_zone` for rising edge |
| `line_crossing` | Segment `(prev, curr)` crosses line | `_line_cross_pending` suppresses repeat until uncross |
| `occupancy_count` | Count of inside centroids | `_last_occupancy` suppresses repeats until count changes |

Motion **direction** and **speed** are computed every frame but are **not** currently used as event triggers in `EventEngine` — they are available for visualization and extension.
