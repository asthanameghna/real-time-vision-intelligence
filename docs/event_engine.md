# Event Engine

The event engine (`app/core/events.py`) evaluates **rule-based** conditions on **2D track centroids** and short **trajectory segments**, using geometry and a small amount of **cross-frame state**. It is intentionally deterministic and side-effect-free aside from updating internal dictionaries.

---

## Configuration Loading

`load_zone_specs(data)` expects a YAML-loaded `dict` with:

| Key | Type | Notes |
|-----|------|------|
| `zones` | list of dicts | Each: `id`, optional `name`, `polygon` as ≥ 3 points `[x, y]`. |
| `lines` | list of dicts | Each: `id`, optional `name`, `p1`, `p2`. |
| `occupancy` | optional dict | If present with `zone_ids`: list of zone IDs to monitor; `null` → empty list. **If `occupancy` is omitted entirely**, `occupancy_zone_ids` is `None` and the engine defaults to **all** zone IDs. |

Points are stored as `Tuple[float, float]`.

---

## Polygon Zone Logic

**Point-in-polygon** uses OpenCV:

```python
r = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), measureDist=False)
return r >= 0.0
```

- `measureDist=False` → returns `+1` inside, `0` on edge, `-1` outside (floating boundary behavior depends on OpenCV’s fixed-point contour handling).
- Polygons are treated as **closed** for drawing (`draw_zones_and_lines` uses `isClosed=True`); inclusion test uses the vertex list order as supplied (typically clockwise or CCW; orientation must be consistent for “inside” to match your intent).

**Zone entry events** are **rising-edge** (enter) only, not exit:

```python
if inside and not was_inside:
    emit zone_entry
self._was_inside_zone[key] = inside
```

There is **no** `zone_exit` event type in the current implementation.

Memory key: `(track_id, zone_id)`. Keys for tracks/zones not seen in the current frame are **removed** after processing (`stale_zone_keys`), so re-entry after the track disappears from the frame can fire `zone_entry` again if the track comes back with the same ID (uncommon) or a new ID (common).

---

## Line Crossing Logic

Crossing uses **segment–segment intersection** between:

- **Motion segment**: `(prev_centroid, curr_centroid)` from the last two points of `trajectory_xy` via `_centroid_pair`.
- **Configured line**: `(p1, p2)`.

The implementation uses a standard **counter-clockwise orientation test** (`_ccw` / `_segments_intersect`) on **open** segments—**endpoint touches** and colinear overlap cases are **not** handled with special-case logic; robust production code often adds epsilon tolerances or “which side of the infinite line” half-plane tests for directionality.

**Deduplication / re-arm**:

- When a crossing is detected and `(track_id, line_id)` is **not** in `_line_cross_pending`, one `line_crossing` event is emitted and the pair is added to the set.
- When **no** intersection is detected this frame, `_line_cross_pending.discard(lk)` — the track must “uncross” (centroid segment no longer intersects) before a **second** crossing can emit.
- If the track **vanishes** from the frame while pending, the pending flag is removed for that `track_id`.

**Direction of crossing** (A→B vs B→A across the line) is **not** encoded in `details` today—only line identity and the two centroid positions.

---

## Occupancy Counting

For each `zone_id` in `self._occupancy_zone_ids`:

1. Resolve `ZoneSpec` from `self._zone_by_id` (unknown IDs are skipped).
2. Count tracks with `track_id >= 0` whose **current** centroid satisfies `_point_in_polygon`.
3. Compare to `_last_occupancy.get(zid)`:
   - If **first time** (`prev_c is None`) or **count changed** → emit `occupancy_count` with `count`, `previous_count` (uses `0` when no previous).

So occupancy is **not** streamed every frame—only on **transitions**, reducing JSONL volume. When the engine starts, the first frame still emits an occupancy event per monitored zone because `prev_c is None`.

**Default occupancy scope**: if YAML omits `occupancy` entirely, `EventEngine` sets `_occupancy_zone_ids` to **every** zone’s `id`. If `occupancy.zone_ids` is an explicit list, only those zones are counted (must match known zone IDs).

---

## Event Deduplication Strategy

| Mechanism | Behavior |
|-----------|----------|
| **Zone entry** | Natural dedupe: only fires on `inside` transition from false→true. |
| **Line crossing** | Pending gate + uncross clears pending; prevents one long intersection from spamming multiple events. |
| **Occupancy** | Emits only when integer count changes (or first sample). |

There is **no** time-based debounce (e.g., “max one event per track per N seconds”) and **no** global event queue deduplication by content hash.

---

## Temporal Event Reasoning

The engine reasons over **two time scales**:

1. **Instantaneous** (this frame): centroid inside polygon? segment intersects line?

2. **One-step temporal** (prev vs curr): line crossing requires `trajectory_xy` length ≥ 2 — effectively **motion between consecutive frames** at the sampling rate implied by `time_sec` steps.

3. **Hysteresis-style** memory: `was_inside`, `line_cross_pending`, `last_occupancy`.

There is **no** higher-order reasoning (e.g., “entered zone A then B within 5s”), no track smoothing, and no confidence gating on events beyond whatever the tracker already filtered.

---

## Event JSON Schema

Each emitted object is a flat JSON-serializable `dict`:

```json
{
  "event_id": "<uuid4 string>",
  "timestamp": <float, seconds on video timeline>,
  "type": "zone_entry" | "line_crossing" | "occupancy_count",
  "track_id": <int or null>,
  "label": "<COCO class name string>",
  "details": { }
}
```

Built in `_new_event()`:

| Field | Type | Notes |
|-------|------|------|
| `event_id` | `str` | `uuid.uuid4()` — unique per emission, not deterministic across runs. |
| `timestamp` | `float` | From `run_detection.py`: `frame_idx / fps`. |
| `type` | `str` | Event discriminator. |
| `track_id` | `int \| None` | `None` for `occupancy_count` (aggregate event). |
| `label` | `str` | Per-object label for track-scoped events; empty string for occupancy. |
| `details` | `dict` | Type-specific payload (below). |

### `zone_entry` — `details`

| Key | Type | Description |
|-----|------|-------------|
| `zone_id` | `str` | Zone identifier from YAML. |
| `zone_name` | `str` | Human-readable name. |
| `centroid` | `[float, float]` | Current centroid `cx, cy`. |

### `line_crossing` — `details`

| Key | Type | Description |
|-----|------|-------------|
| `line_id` | `str` | Line identifier. |
| `line_name` | `str` | Display name. |
| `previous_centroid` | `[float, float]` | Centroid one frame ago (in trajectory). |
| `current_centroid` | `[float, float]` | Centroid this frame. |

### `occupancy_count` — `details`

| Key | Type | Description |
|-----|------|-------------|
| `zone_id` | `str` | |
| `zone_name` | `str` | |
| `count` | `int` | Number of valid tracks inside. |
| `previous_count` | `int` | Last emitted count; `0` if none. |

**Serialization**: one JSON object per line (JSONL), UTF-8, written in frame order.
