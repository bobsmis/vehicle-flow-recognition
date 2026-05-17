"""Member 3: vehicle ID tracking and trajectory association."""

from __future__ import annotations

import math

from .config import DetectionParams
from .models import BBox, Detection, Track


class CentroidTracker:
    def __init__(self, params: DetectionParams) -> None:
        normalized = params.normalized()
        self.max_distance = normalized.max_match_distance
        self.max_lost_frames = normalized.max_lost_frames
        self.history_size = normalized.track_history
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    @property
    def active_tracks(self) -> list[Track]:
        return [track for track in self._tracks.values() if track.lost_frames == 0]

    def update(self, detections: list[Detection]) -> list[Track]:
        if not self._tracks:
            for detection in detections:
                self._create_track(detection)
            return self.active_tracks

        pairs: list[tuple[float, float, int, int]] = []
        track_ids = list(self._tracks.keys())
        for track_id in track_ids:
            track = self._tracks[track_id]
            predicted = track.predicted_centroid
            for detection_index, detection in enumerate(detections):
                distance = math.dist(predicted, detection.centroid)
                if distance <= self.max_distance:
                    iou = bbox_iou(track.bbox, detection.bbox)
                    pairs.append((distance, -iou, track_id, detection_index))

        pairs.sort(key=lambda item: (item[0], item[1]))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for _, _negative_iou, track_id, detection_index in pairs:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            self._tracks[track_id].update(detection.bbox, detection.centroid, self.history_size)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self._tracks[track_id].mark_lost()

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._create_track(detection)

        self._remove_stale_tracks()
        return self.active_tracks

    def _create_track(self, detection: Detection) -> None:
        track = Track(
            track_id=self._next_id,
            bbox=detection.bbox,
            centroid=detection.centroid,
            history=[detection.centroid],
            lost_frames=0,
            age=1,
        )
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _remove_stale_tracks(self) -> None:
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.lost_frames > self.max_lost_frames
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union
