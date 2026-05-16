from app.core.events import _point_in_polygon


def test_point_inside_polygon_returns_true():
    polygon = ((0, 0), (10, 0), (10, 10), (0, 10))

    assert _point_in_polygon((5, 5), polygon) is True


def test_point_outside_polygon_returns_false():
    polygon = ((0, 0), (10, 0), (10, 10), (0, 10))

    assert _point_in_polygon((15, 5), polygon) is False
