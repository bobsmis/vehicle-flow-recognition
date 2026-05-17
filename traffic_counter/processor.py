"""Video processing pipeline and exporters."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import cv2

from .config import AppConfig
from .counter import LineCounter
from .detector import MotionDetector
from .member5_evaluation import evaluate_count_accuracy
from .tracker import CentroidTracker
from .visualization import draw_overlay


@dataclass(slots=True)
class ProcessingState:
    frame_index: int
    total_frames: int
    fps: float
    active_tracks: int
    counts: dict[str, int]
    output_dir: str
    annotated_video: str
    events_in_frame: int
    elapsed_sec: float = 0.0
    processing_fps: float = 0.0
    done: bool = False
    stopped: bool = False
    error: str | None = None


PreviewCallback = Callable[[object, ProcessingState], None]


class VideoProcessor:
    def __init__(self, video_path: str, config: AppConfig) -> None:
        self.video_path = Path(video_path)
        self.config = config.normalized()
        self.output_dir = Path(self.config.output_dir)

    def process(self, callback: PreviewCallback | None, stop_event: Event, pause_event: Event) -> ProcessingState:
        started_at = time.time()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("视频尺寸无效，无法处理。")

        session_dir = self._make_session_dir()
        event_csv_path = session_dir / "过线记录.csv"
        summary_csv_path = session_dir / "统计汇总.csv"
        summary_json_path = session_dir / "统计汇总.json"
        annotated_path = session_dir / f"{self.video_path.stem}_annotated.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"无法创建输出视频：{annotated_path}")

        detector: MotionDetector | None = None
        tracker = CentroidTracker(self.config.params)
        counter = LineCounter(self.config.roi_points, self.config.line_points)
        last_state = ProcessingState(0, total_frames, fps, 0, counter.counts(), str(session_dir), str(annotated_path), 0)

        with event_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            fieldnames = [
                "过线第几辆车",
                "过线时间",
            ]
            writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer_csv.writeheader()

            frame_index = 0
            while not stop_event.is_set():
                while pause_event.is_set() and not stop_event.is_set():
                    time.sleep(0.05)

                ok, frame = cap.read()
                if not ok:
                    break

                if detector is None:
                    detector = MotionDetector(self.config.params, self.config.roi_points, frame.shape)

                frame_index += 1
                timestamp_sec = frame_index / fps if fps > 0 else 0.0
                detections, _ = detector.detect(frame)
                active_tracks = tracker.update(detections)
                events = counter.update(active_tracks, frame_index, timestamp_sec)
                counts = counter.counts()
                annotated = draw_overlay(frame, self.config.roi_points, self.config.line_points, detections, active_tracks, counts, events)
                writer.write(annotated)
                self._write_event_rows(writer_csv, events)

                elapsed = time.time() - started_at
                last_state = ProcessingState(
                    frame_index=frame_index,
                    total_frames=total_frames,
                    fps=fps,
                    active_tracks=len(active_tracks),
                    counts=counts,
                    output_dir=str(session_dir),
                    annotated_video=str(annotated_path),
                    events_in_frame=len(events),
                    elapsed_sec=elapsed,
                    processing_fps=frame_index / elapsed if elapsed > 0 else 0.0,
                )
                if callback is not None:
                    callback(annotated, last_state)

        cap.release()
        writer.release()

        elapsed = time.time() - started_at
        stopped = stop_event.is_set()
        summary = self._build_summary(counter.counts(), total_frames, fps, elapsed, stopped, annotated_path, event_csv_path)
        self._write_summary(summary, summary_csv_path, summary_json_path)

        last_state.done = not stopped
        last_state.stopped = stopped
        last_state.output_dir = str(session_dir)
        last_state.annotated_video = str(annotated_path)
        if callback is not None:
            callback(None, last_state)
        return last_state

    def _make_session_dir(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / f"{self.video_path.stem}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _write_event_rows(
        self,
        writer_csv: csv.DictWriter,
        events,
    ) -> None:
        for event in events:
            writer_csv.writerow(
                {
                    "过线第几辆车": f"第{event.vehicle_number}辆车",
                    "过线时间": self._format_timestamp(event.timestamp_sec),
                }
            )

    def _build_summary(
        self,
        counts: dict[str, int],
        total_frames: int,
        fps: float,
        elapsed: float,
        stopped: bool,
        annotated_path: Path,
        event_csv_path: Path,
    ) -> dict[str, object]:
        evaluation = evaluate_count_accuracy(counts["total"], self.config.manual_total)

        return {
            "视频文件": self.video_path.name,
            "总车流量": counts["total"],
            "人工总数": evaluation.manual_total,
            "总误差": evaluation.error_total,
            "准确率": None if evaluation.accuracy is None else round(evaluation.accuracy, 4),
            "视频总帧数": total_frames,
            "视频FPS": round(fps, 3),
            "处理耗时秒": round(elapsed, 3),
            "是否中途停止": "是" if stopped else "否",
            "标注视频": str(annotated_path),
            "过线记录CSV": str(event_csv_path),
        }

    @staticmethod
    def _write_summary(summary: dict[str, object], summary_csv_path: Path, summary_json_path: Path) -> None:
        with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        with summary_json_path.open("w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_timestamp(timestamp_sec: float) -> str:
        total_ms = int(round(timestamp_sec * 1000))
        total_seconds, milliseconds = divmod(total_ms, 1000)
        minutes_total, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes_total, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
