"""Centralized construction of perception components for the vision stack."""

from __future__ import annotations

from typing import Optional, Sequence

from app.core.events import EventEngine, LineSpec, ZoneSpec
from app.core.motion import MotionEstimator
from app.core.tracker import ByteTrackTracker


class VisionPipeline:
    """Owns ByteTrack tracking, motion estimation, and rule-based event detection."""

    def __init__(
        self,
        *,
        model_path: str,
        conf_threshold: float,
        max_trajectory_points: int,
        stationary_speed_pps: float,
        zone_specs: Sequence[ZoneSpec],
        line_specs: Sequence[LineSpec],
        occupancy_zone_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._max_trajectory_points = max_trajectory_points
        self._stationary_speed_pps = stationary_speed_pps
        self._zone_specs = list(zone_specs)
        self._line_specs = list(line_specs)
        self._occupancy_zone_ids = occupancy_zone_ids
        self._init_perception_components()

    def _init_perception_components(self) -> None:
        self.tracker = ByteTrackTracker(
            model_path=self._model_path, conf_threshold=self._conf_threshold
        )
        self.motion = MotionEstimator(
            max_trajectory_points=self._max_trajectory_points,
            stationary_speed_pps=self._stationary_speed_pps,
        )
        self.event_engine = EventEngine(
            self._zone_specs,
            self._line_specs,
            occupancy_zone_ids=self._occupancy_zone_ids,
        )

    def process_frame(self, frame, time_sec: float = 0.0) -> dict:
        tracks = self.tracker.track(frame)
        motion_states = self.motion.update(tracks, time_sec=time_sec)
        events = self.event_engine.process_frame(tracks, motion_states, time_sec)
        return {
            "tracks": tracks,
            "motion": motion_states,
            "events": events,
            "metrics": {
                "active_tracks": len(tracks),
                "events_count": len(events),
            },
        }

    def reset_state(self) -> None:
        """Recreate tracker, motion, and event state (e.g. when the input video loops)."""
        self._init_perception_components()
