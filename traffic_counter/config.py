"""Configuration dataclasses and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DetectionParams:
    history: int = 500
    var_threshold: float = 32.0
    learning_rate: float = -1.0
    blur_size: int = 5
    morph_kernel: int = 5
    min_area: int = 600
    max_area: int = 60000
    min_rectangularity: float = 0.25
    min_circularity: float = 0.03
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 5.0
    max_match_distance: float = 90.0
    max_lost_frames: int = 15
    track_history: int = 40

    def normalized(self) -> "DetectionParams":
        data = asdict(self)
        data["history"] = max(1, int(data["history"]))
        data["var_threshold"] = max(1.0, float(data["var_threshold"]))
        data["blur_size"] = _odd_at_least(int(data["blur_size"]), 1)
        data["morph_kernel"] = _odd_at_least(int(data["morph_kernel"]), 1)
        data["min_area"] = max(1, int(data["min_area"]))
        data["max_area"] = max(data["min_area"], int(data["max_area"]))
        data["min_rectangularity"] = _clamp(float(data["min_rectangularity"]), 0.0, 1.0)
        data["min_circularity"] = _clamp(float(data["min_circularity"]), 0.0, 1.0)
        data["min_aspect_ratio"] = max(0.01, float(data["min_aspect_ratio"]))
        data["max_aspect_ratio"] = max(data["min_aspect_ratio"], float(data["max_aspect_ratio"]))
        data["max_match_distance"] = max(1.0, float(data["max_match_distance"]))
        data["max_lost_frames"] = max(0, int(data["max_lost_frames"]))
        data["track_history"] = max(2, int(data["track_history"]))
        return DetectionParams(**data)


@dataclass
class AppConfig:
    roi_points: list[list[int]] = field(default_factory=list)
    line_points: list[list[int]] = field(default_factory=list)
    output_dir: str = "output"
    manual_total: int | None = None
    params: DetectionParams = field(default_factory=DetectionParams)

    def normalized(self) -> "AppConfig":
        return AppConfig(
            roi_points=_clean_points(self.roi_points),
            line_points=_clean_points(self.line_points)[:2],
            output_dir=str(self.output_dir or "output"),
            manual_total=_optional_int(self.manual_total),
            params=self.params.normalized(),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self.normalized())
        data["params"] = asdict(self.normalized().params)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        params = DetectionParams(**data.get("params", {})).normalized()
        return cls(
            roi_points=data.get("roi_points", []),
            line_points=data.get("line_points", []),
            output_dir=data.get("output_dir", "output"),
            manual_total=_optional_int(data.get("manual_total")),
            params=params,
        ).normalized()

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)


def _clean_points(points: list[Any]) -> list[list[int]]:
    cleaned: list[list[int]] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            cleaned.append([int(point[0]), int(point[1])])
    return cleaned


def _optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(minimum, value)
    if value % 2 == 0:
        value += 1
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
