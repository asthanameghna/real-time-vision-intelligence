from types import SimpleNamespace

from app.core.events import EventEngine, LineSpec, ZoneSpec, _point_in_polygon


def test_point_inside_polygon_returns_true():
    polygon = ((0, 0), (10, 0), (10, 10), (0, 10))

    assert _point_in_polygon((5, 5), polygon) is True


def test_point_outside_polygon_returns_false():
    polygon = ((0, 0), (10, 0), (10, 10), (0, 10))

    assert _point_in_polygon((15, 5), polygon) is False


def test_event_engine_emits_zone_entry_for_track_inside_zone():
    zone = ZoneSpec(id="zone-1", name="Zone 1", polygon=((0, 0), (10, 0), (10, 10), (0, 10)))
    engine = EventEngine([zone], [], occupancy_zone_ids=[])
    track = SimpleNamespace(track_id=1, label="person", centroid=(5, 5))

    events = engine.process_frame([track], {}, time_sec=1.0)

    assert events[0]["type"] == "zone_entry"


def test_event_engine_deduplicates_zone_entry_while_track_stays_inside_zone():
    zone = ZoneSpec(id="zone-1", name="Zone 1", polygon=((0, 0), (10, 0), (10, 10), (0, 10)))
    engine = EventEngine([zone], [], occupancy_zone_ids=[])
    track = SimpleNamespace(track_id=1, label="person", centroid=(5, 5))

    first_events = engine.process_frame([track], {}, time_sec=1.0)
    second_events = engine.process_frame([track], {}, time_sec=2.0)

    assert [event["type"] for event in first_events] == ["zone_entry"]
    assert second_events == []


def test_event_engine_emits_zone_entry_again_when_track_reenters_zone():
    zone = ZoneSpec(id="zone-1", name="Zone 1", polygon=((0, 0), (10, 0), (10, 10), (0, 10)))
    engine = EventEngine([zone], [], occupancy_zone_ids=[])
    inside_track = SimpleNamespace(track_id=1, label="person", centroid=(5, 5))
    outside_track = SimpleNamespace(track_id=1, label="person", centroid=(15, 5))

    first_events = engine.process_frame([inside_track], {}, time_sec=1.0)
    outside_events = engine.process_frame([outside_track], {}, time_sec=2.0)
    reentry_events = engine.process_frame([inside_track], {}, time_sec=3.0)

    assert [event["type"] for event in first_events] == ["zone_entry"]
    assert outside_events == []
    assert [event["type"] for event in reentry_events] == ["zone_entry"]


def test_event_engine_emits_occupancy_count_for_track_inside_zone():
    zone = ZoneSpec(id="zone-1", name="Zone 1", polygon=((0, 0), (10, 0), (10, 10), (0, 10)))
    engine = EventEngine([zone], [], occupancy_zone_ids=["zone-1"])
    track = SimpleNamespace(track_id=1, label="person", centroid=(5, 5))

    events = engine.process_frame([track], {}, time_sec=1.0)
    occupancy_event = [event for event in events if event["type"] == "occupancy_count"][0]

    assert occupancy_event["details"]["count"] == 1
    assert occupancy_event["details"]["previous_count"] == 0


def test_event_engine_emits_occupancy_count_when_count_decreases():
    zone = ZoneSpec(id="zone-1", name="Zone 1", polygon=((0, 0), (10, 0), (10, 10), (0, 10)))
    engine = EventEngine([zone], [], occupancy_zone_ids=["zone-1"])
    track = SimpleNamespace(track_id=1, label="person", centroid=(5, 5))

    engine.process_frame([track], {}, time_sec=1.0)
    events = engine.process_frame([], {}, time_sec=2.0)
    occupancy_event = [event for event in events if event["type"] == "occupancy_count"][0]

    assert occupancy_event["details"]["count"] == 0
    assert occupancy_event["details"]["previous_count"] == 1


def test_event_engine_emits_line_crossing_when_track_crosses_line():
    line = LineSpec(id="line-1", name="Line 1", p1=(5, 0), p2=(5, 10))
    engine = EventEngine([], [line])
    first_track = SimpleNamespace(track_id=1, label="person", centroid=(0, 5))
    second_track = SimpleNamespace(track_id=1, label="person", centroid=(10, 5))

    engine.process_frame([first_track], {1: SimpleNamespace(trajectory_xy=((0, 5),))}, time_sec=1.0)
    events = engine.process_frame(
        [second_track],
        {1: SimpleNamespace(trajectory_xy=((0, 5), (10, 5)))},
        time_sec=2.0,
    )

    assert [event["type"] for event in events] == ["line_crossing"]
    