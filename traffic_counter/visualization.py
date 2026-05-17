"""Frame annotation helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import CountEvent, Detection, Track

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def draw_overlay(
    frame: np.ndarray,
    roi_points: list[list[int]],
    line_points: list[list[int]],
    detections: list[Detection],
    tracks: list[Track],
    counts: dict[str, int],
    events: list[CountEvent] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    _draw_roi(annotated, roi_points)
    _draw_line(annotated, line_points)
    _draw_detections(annotated, detections)
    _draw_tracks(annotated, tracks)
    _draw_counts(annotated, counts)
    if events:
        _draw_events(annotated, events)
    return annotated


def draw_setup_overlay(frame: np.ndarray, roi_points: list[list[int]], line_points: list[list[int]]) -> np.ndarray:
    annotated = frame.copy()
    _draw_roi(annotated, roi_points)
    _draw_line(annotated, line_points)
    return annotated


def _draw_roi(frame: np.ndarray, roi_points: list[list[int]]) -> None:
    if len(roi_points) < 2:
        for point in roi_points:
            cv2.circle(frame, tuple(point), 5, (0, 255, 255), -1)
        return

    points = np.array(roi_points, dtype=np.int32)
    closed = len(roi_points) >= 3
    cv2.polylines(frame, [points], closed, (0, 255, 255), 2)
    if closed:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (0, 140, 140))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for index, point in enumerate(roi_points, start=1):
        cv2.circle(frame, tuple(point), 4, (0, 255, 255), -1)
        cv2.putText(frame, str(index), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def _draw_line(frame: np.ndarray, line_points: list[list[int]]) -> None:
    if not line_points:
        return
    for point in line_points:
        cv2.circle(frame, tuple(point), 5, (255, 0, 255), -1)
    if len(line_points) >= 2:
        start = tuple(line_points[0])
        end = tuple(line_points[1])
        cv2.arrowedLine(frame, start, end, (255, 0, 255), 3, tipLength=0.08)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        _draw_pil_text(frame, "计数线", (mid[0] + 8, mid[1] - 24), 18, (255, 0, 255))


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 180, 255), 1)


def _draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    for track in tracks:
        x, y, w, h = track.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.circle(frame, track.centroid, 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {track.track_id}", (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        if len(track.history) >= 2:
            for p1, p2 in zip(track.history[:-1], track.history[1:]):
                cv2.line(frame, p1, p2, (0, 200, 255), 2)


def _draw_counts(frame: np.ndarray, counts: dict[str, int]) -> None:
    cv2.rectangle(frame, (10, 10), (180, 48), (0, 0, 0), -1)
    _draw_pil_text(frame, f"总数：{counts.get('total', 0)}", (20, 18), 22, (80, 220, 255))


def _draw_events(frame: np.ndarray, events: list[CountEvent]) -> None:
    for event in events:
        cv2.circle(frame, event.center, 12, (0, 255, 0), 3)


def _draw_pil_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int, color_bgr: tuple[int, int, int]) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(xy, text, font=_get_font(size), fill=color_rgb)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _get_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arialuni.ttf",
    ]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font
