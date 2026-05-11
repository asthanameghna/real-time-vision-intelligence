"""Ultralytics YOLO + ByteTrack multi-object tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# Resolved via Ultralytics package cfg (see ultralytics/cfg/trackers/bytetrack.yaml)
TRACKER_CFG = "bytetrack.yaml"


@dataclass
class TrackedObject:
    """One tracked detection after association."""

    track_id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]


class ByteTrackTracker:
    """Frame-by-frame tracking using YOLO detect + ByteTrack (Ultralytics built-in)."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
    ) -> None:
        self._model = YOLO(model_path)
        self._conf_threshold = conf_threshold

    def track(self, frame: np.ndarray) -> List[TrackedObject]:
        """Run tracking on one BGR frame; call in temporal order with the same instance."""
        results = self._model.track(
            source=frame,
            conf=self._conf_threshold,
            persist=True,
            tracker=TRACKER_CFG,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: List[TrackedObject] = []
        for d in boxes:
            xyxy = d.xyxy.squeeze().cpu().numpy().reshape(4)
            x1, y1, x2, y2 = map(float, xyxy)
            conf = float(d.conf.item())
            cls_id = int(d.cls.item())
            label = str(names.get(cls_id, str(cls_id)))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if d.is_track:
                tid = int(d.id.item())
            else:
                tid = -1
            out.append(
                TrackedObject(
                    track_id=tid,
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    centroid=(cx, cy),
                )
            )
        return out
