"""Member 2: vehicle candidate detection from foreground contours."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import DetectionParams
from .models import Detection


class VehicleContourDetector:
    """Find foreground contours and keep only vehicle-like shapes."""

    def __init__(self, params: DetectionParams) -> None:
        self.params = params.normalized()

    def detect(self, fg_mask: np.ndarray) -> list[Detection]:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            detection = self._contour_to_detection(contour)
            if detection is not None:
                detections.append(detection)

        detections.sort(key=lambda item: item.area, reverse=True)
        return detections

    def _contour_to_detection(self, contour: np.ndarray) -> Detection | None:
        area = float(cv2.contourArea(contour))
        if area < self.params.min_area or area > self.params.max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None

        rect_area = float(w * h)
        rectangularity = area / rect_area if rect_area else 0.0
        aspect_ratio = w / float(h)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

        if rectangularity < self.params.min_rectangularity:
            return None
        if circularity < self.params.min_circularity:
            return None
        if not (self.params.min_aspect_ratio <= aspect_ratio <= self.params.max_aspect_ratio):
            return None

        centroid = (int(x + w / 2), int(y + h / 2))
        return Detection(
            bbox=(int(x), int(y), int(w), int(h)),
            centroid=centroid,
            area=area,
            rectangularity=rectangularity,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            contour=contour,
        )
