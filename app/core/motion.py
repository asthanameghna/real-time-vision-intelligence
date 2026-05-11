"""Trajectory recording and motion estimation per track."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Deque, Dict, List, Tuple

from app.core.tracker import TrackedObject


class MotionDirection(str, Enum):
    """Dominant direction from recent velocity (image coordinates: +y is down)."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    STATIONARY = "stationary"


@dataclass(frozen=True)
class TrackMotionState:
    """Motion and path for one track after an update."""

    track_id: int
    trajectory_xy: Tuple[Tuple[float, float], ...]
    speed_pps: float
    direction: MotionDirection
    velocity_xy: Tuple[float, float]


def _direction_from_velocity(
    vx: float, vy: float, speed: float, stationary_speed: float
) -> MotionDirection:
    if speed < stationary_speed:
        return MotionDirection.STATIONARY
    if abs(vx) >= abs(vy):
        return MotionDirection.LEFT if vx < 0 else MotionDirection.RIGHT
    return MotionDirection.UP if vy < 0 else MotionDirection.DOWN


@dataclass
class _Sample:
    t: float
    x: float
    y: float


class MotionEstimator:
    """Centroid trajectory and velocity/direction per track_id."""

    def __init__(
        self,
        max_trajectory_points: int = 64,
        stationary_speed_pps: float = 12.0,
    ) -> None:
        if max_trajectory_points < 2:
            raise ValueError("max_trajectory_points must be at least 2")
        self._max_trajectory_points = max_trajectory_points
        self._stationary_speed_pps = stationary_speed_pps
        self._histories: Dict[int, Deque[_Sample]] = {}

    def update(
        self, tracks: List[TrackedObject], time_sec: float
    ) -> Dict[int, TrackMotionState]:
        """Append centroids for this frame; return motion state for each active track."""
        active_ids: set[int] = set()
        out: Dict[int, TrackMotionState] = {}

        for t in tracks:
            if t.track_id < 0:
                continue
            active_ids.add(t.track_id)
            hist = self._histories.get(t.track_id)
            if hist is None:
                hist = deque(maxlen=self._max_trajectory_points)
                self._histories[t.track_id] = hist
            cx, cy = t.centroid
            hist.append(_Sample(t=time_sec, x=cx, y=cy))

            traj = tuple((s.x, s.y) for s in hist)
            vx, vy = 0.0, 0.0
            speed = 0.0
            if len(hist) >= 2:
                s0, s1 = hist[-2], hist[-1]
                dt = s1.t - s0.t
                if dt > 1e-9:
                    vx = (s1.x - s0.x) / dt
                    vy = (s1.y - s0.y) / dt
                    speed = math.hypot(vx, vy)

            direction = _direction_from_velocity(
                vx, vy, speed, self._stationary_speed_pps
            )
            out[t.track_id] = TrackMotionState(
                track_id=t.track_id,
                trajectory_xy=traj,
                speed_pps=speed,
                direction=direction,
                velocity_xy=(vx, vy),
            )

        stale = [tid for tid in self._histories if tid not in active_ids]
        for tid in stale:
            del self._histories[tid]

        return out
