"""Member 4: virtual line crossing and duplicate-free traffic counting."""

from __future__ import annotations

from .geometry import as_points, intersection_point, point_in_polygon, segments_intersect, signed_side
from .models import CountEvent, Track


class LineCounter:
    def __init__(self, roi_points: list[list[int]], line_points: list[list[int]]) -> None:
        self.roi = as_points(roi_points)
        self.line = as_points(line_points)[:2]
        self._total_count = 0
        self.counted_track_ids: set[int] = set()

    @property
    def total_count(self) -> int:
        return self._total_count

    def update(self, tracks: list[Track], frame_index: int, timestamp_sec: float) -> list[CountEvent]:
        if len(self.line) != 2:
            return []

        events: list[CountEvent] = []
        line_start, line_end = self.line
        for track in tracks:
            if track.track_id in self.counted_track_ids or len(track.history) < 2:
                continue

            previous = track.history[-2]
            current = track.history[-1]
            previous_side = signed_side(line_start, line_end, previous)
            current_side = signed_side(line_start, line_end, current)
            direction = crossing_direction(previous_side, current_side)
            if direction == 0:
                continue
            if not segments_intersect(previous, current, line_start, line_end):
                continue

            crossing = intersection_point(previous, current, line_start, line_end)
            if crossing is not None and not point_in_polygon(crossing, self.roi):
                continue

            self._total_count += 1
            self.counted_track_ids.add(track.track_id)
            events.append(
                CountEvent(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    vehicle_number=self.total_count,
                    track_id=track.track_id,
                    center=current,
                )
            )

        return events

    def counts(self) -> dict[str, int]:
        return {
            "total": self.total_count,
        }


def crossing_direction(previous_side: float, current_side: float) -> int:
    if previous_side == current_side:
        return 0
    if previous_side * current_side > 0:
        return 0
    return 1 if current_side > previous_side else -1
