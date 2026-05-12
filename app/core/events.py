"""Rule-based event detection from track centroids and zone/line geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import uuid

import cv2
import numpy as np

from app.core.motion import TrackMotionState
from app.core.tracker import TrackedObject


Point = Tuple[float, float]


@dataclass(frozen=True)
class ZoneSpec:
    id: str
    name: str
    polygon: Tuple[Point, ...]


@dataclass(frozen=True)
class LineSpec:
    id: str
    name: str
    p1: Point
    p2: Point


def _as_point(v: Any) -> Point:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"Expected [x, y], got {v!r}")
    return (float(v[0]), float(v[1]))


def load_zone_specs(
    data: Mapping[str, Any],
) -> Tuple[List[ZoneSpec], List[LineSpec], Optional[List[str]]]:
    """Parse zones, lines, and optional occupancy.zone_ids from a YAML-loaded mapping."""
    zones_raw = data.get("zones") or []
    lines_raw = data.get("lines") or []
    if not isinstance(zones_raw, list):
        raise ValueError("'zones' must be a list")
    if not isinstance(lines_raw, list):
        raise ValueError("'lines' must be a list")

    zones: List[ZoneSpec] = []
    for z in zones_raw:
        if not isinstance(z, dict):
            raise ValueError("Each zone must be a mapping")
        zid = str(z["id"])
        name = str(z.get("name", zid))
        poly = z.get("polygon") or []
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError(f"Zone {zid!r}: polygon must have at least 3 vertices")
        pts = tuple(_as_point(p) for p in poly)
        zones.append(ZoneSpec(id=zid, name=name, polygon=pts))

    lines: List[LineSpec] = []
    for ln in lines_raw:
        if not isinstance(ln, dict):
            raise ValueError("Each line must be a mapping")
        lid = str(ln["id"])
        name = str(ln.get("name", lid))
        p1 = _as_point(ln["p1"])
        p2 = _as_point(ln["p2"])
        lines.append(LineSpec(id=lid, name=name, p1=p1, p2=p2))

    occ_ids: Optional[List[str]] = None
    occ = data.get("occupancy")
    if isinstance(occ, dict) and "zone_ids" in occ:
        raw_ids = occ["zone_ids"]
        if raw_ids is None:
            occ_ids = []
        elif not isinstance(raw_ids, list):
            raise ValueError("occupancy.zone_ids must be a list or null")
        else:
            occ_ids = [str(x) for x in raw_ids]

    return zones, lines, occ_ids


def _point_in_polygon(pt: Point, polygon: Sequence[Point]) -> bool:
    poly = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    r = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
    return r >= 0.0


def _ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _centroid_pair(
    motion_by_id: Mapping[int, TrackMotionState], track_id: int
) -> Optional[Tuple[Point, Point]]:
    m = motion_by_id.get(track_id)
    if m is None or len(m.trajectory_xy) < 2:
        return None
    prev = m.trajectory_xy[-2]
    curr = m.trajectory_xy[-1]
    return (prev, curr)


class EventEngine:
    """Emit zone_entry, line_crossing, and occupancy_count events from track centroids."""

    def __init__(
        self,
        zones: Sequence[ZoneSpec],
        lines: Sequence[LineSpec],
        *,
        occupancy_zone_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self._zones = list(zones)
        self._lines = list(lines)
        if occupancy_zone_ids is None:
            self._occupancy_zone_ids = {z.id for z in self._zones}
        else:
            self._occupancy_zone_ids = set(str(x) for x in occupancy_zone_ids)
        self._zone_by_id: Dict[str, ZoneSpec] = {z.id: z for z in self._zones}

        # (track_id, zone_id) -> was inside last frame (for zone_entry dedupe)
        self._was_inside_zone: Dict[Tuple[int, str], bool] = {}
        # (track_id, line_id) pending clear after a crossing emit
        self._line_cross_pending: Set[Tuple[int, str]] = set()
        self._last_occupancy: Dict[str, int] = {}

    def _new_event(
        self,
        *,
        time_sec: float,
        event_type: str,
        track_id: Optional[int],
        label: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": float(time_sec),
            "type": event_type,
            "track_id": track_id,
            "label": label,
            "details": details,
        }

    def process_frame(
        self,
        tracks: List[TrackedObject],
        motion_by_id: Mapping[int, TrackMotionState],
        time_sec: float,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        active_pairs: Set[Tuple[int, str]] = set()

        for t in tracks:
            if t.track_id < 0:
                continue
            cx, cy = t.centroid
            pt: Point = (cx, cy)

            for zone in self._zones:
                key = (t.track_id, zone.id)
                active_pairs.add(key)
                inside = _point_in_polygon(pt, zone.polygon)
                was_inside = self._was_inside_zone.get(key, False)
                if inside and not was_inside:
                    events.append(
                        self._new_event(
                            time_sec=time_sec,
                            event_type="zone_entry",
                            track_id=t.track_id,
                            label=t.label,
                            details={
                                "zone_id": zone.id,
                                "zone_name": zone.name,
                                "centroid": [cx, cy],
                            },
                        )
                    )
                self._was_inside_zone[key] = inside

            for line in self._lines:
                pair = _centroid_pair(motion_by_id, t.track_id)
                lk = (t.track_id, line.id)
                if pair is None:
                    continue
                prev, curr = pair
                crosses = _segments_intersect(prev, curr, line.p1, line.p2)
                if crosses:
                    if lk not in self._line_cross_pending:
                        events.append(
                            self._new_event(
                                time_sec=time_sec,
                                event_type="line_crossing",
                                track_id=t.track_id,
                                label=t.label,
                                details={
                                    "line_id": line.id,
                                    "line_name": line.name,
                                    "previous_centroid": [prev[0], prev[1]],
                                    "current_centroid": [curr[0], curr[1]],
                                },
                            )
                        )
                        self._line_cross_pending.add(lk)
                else:
                    self._line_cross_pending.discard(lk)

        active_tids = {t.track_id for t in tracks if t.track_id >= 0}
        for lk in list(self._line_cross_pending):
            if lk[0] not in active_tids:
                self._line_cross_pending.discard(lk)

        stale_zone_keys = [k for k in self._was_inside_zone if k not in active_pairs]
        for k in stale_zone_keys:
            del self._was_inside_zone[k]

        for zid in self._occupancy_zone_ids:
            zone = self._zone_by_id.get(zid)
            if zone is None:
                continue
            count = 0
            for t in tracks:
                if t.track_id < 0:
                    continue
                if _point_in_polygon(t.centroid, zone.polygon):
                    count += 1
            prev_c = self._last_occupancy.get(zid)
            if prev_c is None or prev_c != count:
                events.append(
                    self._new_event(
                        time_sec=time_sec,
                        event_type="occupancy_count",
                        track_id=None,
                        label="",
                        details={
                            "zone_id": zone.id,
                            "zone_name": zone.name,
                            "count": count,
                            "previous_count": prev_c if prev_c is not None else 0,
                        },
                    )
                )
                self._last_occupancy[zid] = count

        return events


def draw_zones_and_lines(
    frame: np.ndarray,
    zones: Sequence[ZoneSpec],
    lines: Sequence[LineSpec],
    *,
    zone_bgr: Tuple[int, int, int] = (80, 220, 80),
    line_bgr: Tuple[int, int, int] = (64, 128, 255),
) -> None:
    """Draw closed zone polygons and line segments on a BGR frame (in place)."""
    for z in zones:
        pts = np.array(z.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=zone_bgr,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    for ln in lines:
        p1 = (int(round(ln.p1[0])), int(round(ln.p1[1])))
        p2 = (int(round(ln.p2[0])), int(round(ln.p2[1])))
        cv2.line(frame, p1, p2, line_bgr, 3, cv2.LINE_AA)


def draw_recent_event_alerts(
    frame: np.ndarray,
    recent: Sequence[Mapping[str, Any]],
    *,
    max_lines: int = 8,
) -> None:
    """Draw a short stack of recent event summaries (newest at the top)."""
    if not recent:
        return
    w = frame.shape[1]
    items = list(recent)[-max_lines:]
    text_lines: List[str] = []
    for ev in reversed(items):
        et = str(ev.get("type", ""))
        ts = float(ev.get("timestamp", 0.0))
        tid = ev.get("track_id")
        label = str(ev.get("label", ""))
        parts = [et, f"t={ts:.2f}s", f"id={tid if tid is not None else 'n/a'}"]
        if label:
            parts.append(label[:20])
        text_lines.append(" | ".join(parts))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    text_h = 16
    pad = 6
    box_w = 200
    for s in text_lines:
        (tw, _th), _ = cv2.getTextSize(s, font, scale, thick)
        box_w = max(box_w, tw + 2 * pad)
    box_w = min(box_w, w - 12)
    box_h = len(text_lines) * text_h + 2 * pad

    x0, y_top = 8, 10
    cv2.rectangle(
        frame,
        (x0, y_top),
        (x0 + box_w, y_top + box_h),
        (45, 45, 45),
        -1,
    )
    cv2.rectangle(
        frame,
        (x0, y_top),
        (x0 + box_w, y_top + box_h),
        (200, 200, 200),
        1,
    )

    y = y_top + pad + text_h - 3
    for s in text_lines:
        cv2.putText(
            frame,
            s,
            (x0 + pad, y),
            font,
            scale,
            (250, 250, 250),
            thick,
            cv2.LINE_AA,
        )
        y += text_h
