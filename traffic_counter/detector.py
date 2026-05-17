"""Compatibility facade for member 1 foreground extraction and member 2 detection."""

from __future__ import annotations

import numpy as np

from .config import DetectionParams
from .member1_foreground import ForegroundExtractor
from .member2_vehicle_detection import VehicleContourDetector
from .models import Detection


class MotionDetector:
    def __init__(self, params: DetectionParams, roi_points: list[list[int]], frame_shape: tuple[int, int, int]) -> None:
        self.foreground_extractor = ForegroundExtractor(params, roi_points, frame_shape)
        self.vehicle_detector = VehicleContourDetector(params)
        self.params = self.foreground_extractor.params
        self.roi_points = roi_points
        self.subtractor = self.foreground_extractor.subtractor
        self.roi_mask = self.foreground_extractor.roi_mask

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        fg_mask = self.foreground_extractor.extract(frame)
        detections = self.vehicle_detector.detect(fg_mask)
        return detections, fg_mask
