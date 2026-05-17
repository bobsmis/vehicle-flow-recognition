"""Small geometry helpers used by the line counter."""

from __future__ import annotations

from .models import Point


def signed_side(a: Point, b: Point, p: Point) -> float:
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1 = signed_side(q1, q2, p1)
    d2 = signed_side(q1, q2, p2)
    d3 = signed_side(p1, p2, q1)
    d4 = signed_side(p1, p2, q2)

    if _opposite_or_zero(d1, d2) and _opposite_or_zero(d3, d4):
        return True
    return False


def intersection_point(p1: Point, p2: Point, q1: Point, q2: Point) -> tuple[float, float] | None:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return px, py


def point_in_polygon(point: tuple[float, float], polygon: list[Point]) -> bool:
    if len(polygon) < 3:
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def as_points(points: list[list[int]]) -> list[Point]:
    return [(int(point[0]), int(point[1])) for point in points if len(point) >= 2]


def _opposite_or_zero(a: float, b: float) -> bool:
    eps = 1e-6
    return (a <= eps and b >= -eps) or (a >= -eps and b <= eps)
