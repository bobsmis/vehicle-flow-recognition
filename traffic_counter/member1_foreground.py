"""Member 1: foreground extraction with OpenCV background modeling."""

from __future__ import annotations

import cv2
import numpy as np

from .config import DetectionParams


class ForegroundExtractor:
    """Build a clean moving-object foreground mask for each video frame."""

    def __init__(self, params: DetectionParams, roi_points: list[list[int]], frame_shape: tuple[int, int, int]) -> None:
        self.params = params.normalized()
        self.roi_points = roi_points
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.params.history,
            varThreshold=self.params.var_threshold,
            detectShadows=True,
        )
        self.roi_mask = self._build_roi_mask(frame_shape)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        processed = frame
        if self.roi_mask is not None:
            processed = cv2.bitwise_and(frame, frame, mask=self.roi_mask)

        if self.params.blur_size > 1:
            processed = cv2.GaussianBlur(processed, (self.params.blur_size, self.params.blur_size), 0)

        fg_mask = self.subtractor.apply(processed, learningRate=self.params.learning_rate)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        if self.roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, self.roi_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.params.morph_kernel, self.params.morph_kernel),
        )
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        return fg_mask

    def _build_roi_mask(self, frame_shape: tuple[int, int, int]) -> np.ndarray | None:
        if len(self.roi_points) < 3:
            return None
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array(self.roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask
