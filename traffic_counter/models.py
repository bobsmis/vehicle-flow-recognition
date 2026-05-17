"""Shared data models for detection, tracking, and counting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class Detection:
    bbox: BBox
    centroid: Point
    area: float
    rectangularity: float
    circularity: float
    aspect_ratio: float
    contour: Any | None = None


@dataclass
class Track:
    track_id: int
    bbox: BBox
    centroid: Point
    history: list[Point] = field(default_factory=list)
    lost_frames: int = 0
    age: int = 0

    def update(self, bbox: BBox, centroid: Point, history_size: int) -> None:
        self.bbox = bbox
        self.centroid = centroid
        self.history.append(centroid)
        if len(self.history) > history_size:
            self.history = self.history[-history_size:]
        self.lost_frames = 0
        self.age += 1

    def mark_lost(self) -> None:
        self.lost_frames += 1
        self.age += 1

    @property
    def velocity(self) -> tuple[float, float]:
        if len(self.history) < 2:
            return 0.0, 0.0
        x1, y1 = self.history[-2]
        x2, y2 = self.history[-1]
        return float(x2 - x1), float(y2 - y1)

    @property
    def predicted_centroid(self) -> tuple[float, float]:
        vx, vy = self.velocity
        return self.centroid[0] + vx, self.centroid[1] + vy


@dataclass(slots=True)
class CountEvent:
    frame_index: int
    timestamp_sec: float
    vehicle_number: int
    track_id: int
    center: Point
