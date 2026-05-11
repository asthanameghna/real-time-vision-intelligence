"""Ultralytics YOLO11 object detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """One bounding box from inference."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


class ObjectDetector:
    """Thin wrapper around a YOLO11 detection model."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.25,
    ) -> None:
        self._model = YOLO(model_path)
        self._conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame (OpenCV convention)."""
        results = self._model.predict(
            source=frame,
            conf=self._conf_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        out: List[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return out

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, str(cls_id)))
            out.append(
                Detection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    label=label,
                )
            )
        return out
