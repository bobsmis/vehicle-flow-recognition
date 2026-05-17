"""Tkinter desktop GUI for the vehicle traffic counter."""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from .config import AppConfig, DetectionParams
from .processor import ProcessingState, VideoProcessor
from .visualization import draw_setup_overlay


class TrafficCounterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("车流量统计系统 - Python + OpenCV")
        self.geometry("1280x760")
        self.minsize(1080, 680)

        self.video_path: str | None = None
        self.config = AppConfig(output_dir=str(Path.cwd() / "output"))
        self.first_frame = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.draw_mode = tk.StringVar(value="roi")

        self.param_vars: dict[str, tk.StringVar] = {}
        self.output_dir_var = tk.StringVar(value=self.config.output_dir)
        self.manual_total_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="请选择视频文件。")
        self.progress_var = tk.StringVar(value="0 / 0")
        self.total_var = tk.StringVar(value="0")
        self.active_tracks_var = tk.StringVar(value="0")
        self.processing_fps_var = tk.StringVar(value="0.0")
        self.elapsed_var = tk.StringVar(value="0.0 秒")
        self.frame_event_var = tk.StringVar(value="0")
        self.output_result_var = tk.StringVar(value="")

        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.result_queue: queue.Queue[tuple[str, object, object | None]] = queue.Queue(maxsize=8)

        self._build_ui()
        self._load_params_to_vars(self.config.params)
        self.after(80, self._poll_worker_queue)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=4)
        paned.add(right, weight=1)

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, bg="#151515", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self._render_setup_frame())

        live_bar = self._build_live_bar(left)
        live_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        status = ttk.Label(left, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self._build_controls(right)

    def _build_live_bar(self, parent: ttk.Frame) -> ttk.Frame:
        live_bar = ttk.LabelFrame(parent, text="实时统计", padding=8)
        for column in range(4):
            live_bar.columnconfigure(column, weight=1)

        items = [
            ("进度", self.progress_var),
            ("总数", self.total_var),
            ("当前目标", self.active_tracks_var),
            ("处理FPS", self.processing_fps_var),
        ]
        for column, (label, var) in enumerate(items):
            cell = ttk.Frame(live_bar)
            cell.grid(row=0, column=column, sticky="ew", padx=4)
            ttk.Label(cell, text=label, anchor="center").pack(fill=tk.X)
            ttk.Label(cell, textvariable=var, anchor="center", font=("Microsoft YaHei", 15, "bold")).pack(fill=tk.X)
        return live_bar

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        file_group = ttk.LabelFrame(parent, text="视频与配置", padding=8)
        file_group.grid(row=0, column=0, sticky="ew")
        file_group.columnconfigure(0, weight=1)
        ttk.Button(file_group, text="选择视频", command=self._select_video).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(file_group, text="保存配置", command=self._save_config).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(file_group, text="加载配置", command=self._load_config).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(file_group, text="选择输出目录", command=self._select_output_dir).grid(row=3, column=0, sticky="ew", pady=2)

        draw_group = ttk.LabelFrame(parent, text="ROI 与计数线", padding=8)
        draw_group.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        draw_group.columnconfigure(0, weight=1)
        ttk.Radiobutton(draw_group, text="绘制 ROI 多边形", variable=self.draw_mode, value="roi").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(draw_group, text="绘制虚拟计数线", variable=self.draw_mode, value="line").grid(row=1, column=0, sticky="w")
        buttons = ttk.Frame(draw_group)
        buttons.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(buttons, text="撤销点", command=self._undo_point).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(buttons, text="清空ROI", command=self._clear_roi).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(buttons, text="清空线", command=self._clear_line).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        params_group = ttk.LabelFrame(parent, text="检测与跟踪参数", padding=8)
        params_group.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        parent.rowconfigure(2, weight=1)
        for row, (attr, label) in enumerate(self._param_specs()):
            ttk.Label(params_group, text=label).grid(row=row, column=0, sticky="w", pady=1)
            var = tk.StringVar()
            self.param_vars[attr] = var
            ttk.Entry(params_group, textvariable=var, width=10).grid(row=row, column=1, sticky="ew", pady=1)
        params_group.columnconfigure(1, weight=1)

        truth_group = ttk.LabelFrame(parent, text="人工真值（可选）", padding=8)
        truth_group.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        truth_group.columnconfigure(1, weight=1)
        ttk.Label(truth_group, text="人工总数").grid(row=0, column=0, sticky="w")
        ttk.Entry(truth_group, textvariable=self.manual_total_var, width=8).grid(row=0, column=1, sticky="ew")

        run_group = ttk.LabelFrame(parent, text="处理", padding=8)
        run_group.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        run_group.columnconfigure((0, 1, 2), weight=1)
        self.start_button = ttk.Button(run_group, text="开始", command=self._start_processing)
        self.pause_button = ttk.Button(run_group, text="暂停", command=self._toggle_pause, state=tk.DISABLED)
        self.stop_button = ttk.Button(run_group, text="停止", command=self._stop_processing, state=tk.DISABLED)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=3)
        self.stop_button.grid(row=0, column=2, sticky="ew", padx=(3, 0))
        ttk.Button(run_group, text="打开输出目录", command=self._open_output_dir).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        stats_group = ttk.LabelFrame(parent, text="统计状态", padding=8)
        stats_group.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        for row, (label, var) in enumerate(
            [
                ("进度", self.progress_var),
                ("总数", self.total_var),
                ("当前目标", self.active_tracks_var),
                ("本帧计数", self.frame_event_var),
                ("处理FPS", self.processing_fps_var),
                ("处理耗时", self.elapsed_var),
                ("输出", self.output_result_var),
            ]
        ):
            ttk.Label(stats_group, text=label).grid(row=row, column=0, sticky="w")
            ttk.Label(stats_group, textvariable=var, wraplength=240).grid(row=row, column=1, sticky="w")

    def _param_specs(self) -> list[tuple[str, str]]:
        return [
            ("history", "背景历史"),
            ("var_threshold", "背景阈值"),
            ("learning_rate", "学习率"),
            ("blur_size", "模糊核"),
            ("morph_kernel", "形态核"),
            ("min_area", "最小面积"),
            ("max_area", "最大面积"),
            ("min_rectangularity", "矩形度"),
            ("min_circularity", "圆形度"),
            ("min_aspect_ratio", "最小宽高比"),
            ("max_aspect_ratio", "最大宽高比"),
            ("max_match_distance", "匹配距离"),
            ("max_lost_frames", "丢失帧数"),
            ("track_history", "轨迹长度"),
        ]

    def _select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择测试视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("无法打开视频", "视频读取失败，请选择其他文件。")
            return

        self.video_path = path
        self.first_frame = frame
        self.config.roi_points = []
        self.config.line_points = []
        self.status_var.set(f"已选择视频：{path}")
        self.progress_var.set("0 / 0")
        self.output_result_var.set("")
        self._reset_counts()
        self._render_setup_frame()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.first_frame is None or self._is_processing():
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            return

        self._sync_config_from_vars()
        if self.draw_mode.get() == "roi":
            self.config.roi_points.append([point[0], point[1]])
            self.status_var.set(f"ROI 点数：{len(self.config.roi_points)}")
        else:
            if len(self.config.line_points) >= 2:
                self.config.line_points = []
            self.config.line_points.append([point[0], point[1]])
            self.status_var.set(f"计数线点数：{len(self.config.line_points)}")
        self._render_setup_frame()

    def _undo_point(self) -> None:
        if self.draw_mode.get() == "line" and self.config.line_points:
            self.config.line_points.pop()
        elif self.config.roi_points:
            self.config.roi_points.pop()
        self._render_setup_frame()

    def _clear_roi(self) -> None:
        self.config.roi_points = []
        self._render_setup_frame()

    def _clear_line(self) -> None:
        self.config.line_points = []
        self._render_setup_frame()

    def _select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir_var.set(path)
            self._sync_config_from_vars()

    def _save_config(self) -> None:
        if self.first_frame is None:
            messagebox.showwarning("未选择视频", "请先选择视频并配置 ROI/计数线。")
            return
        if not self._sync_config_from_vars(show_error=True):
            return
        default_name = "traffic_config.json"
        if self.video_path:
            default_name = f"{Path(self.video_path).stem}_config.json"
        path = filedialog.asksaveasfilename(
            title="保存配置",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON 配置", "*.json")],
        )
        if path:
            self.config.save(path)
            self.status_var.set(f"配置已保存：{path}")

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(title="加载配置", filetypes=[("JSON 配置", "*.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.config = AppConfig.load(path)
        except Exception as exc:
            messagebox.showerror("配置读取失败", str(exc))
            return
        self.output_dir_var.set(self.config.output_dir)
        self.manual_total_var.set("" if self.config.manual_total is None else str(self.config.manual_total))
        self._load_params_to_vars(self.config.params)
        self._render_setup_frame()
        self.status_var.set(f"配置已加载：{path}")

    def _start_processing(self) -> None:
        if self._is_processing():
            return
        if not self.video_path:
            messagebox.showwarning("未选择视频", "请先选择视频文件。")
            return
        if len(self.config.roi_points) < 3:
            messagebox.showwarning("ROI 未完成", "请在首帧上至少绘制 3 个 ROI 点。")
            return
        if len(self.config.line_points) != 2:
            messagebox.showwarning("计数线未完成", "请绘制虚拟计数线的两个端点。")
            return
        if not self._sync_config_from_vars(show_error=True):
            return

        self.stop_event.clear()
        self.pause_event.clear()
        self._reset_counts()
        self.start_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.NORMAL, text="暂停")
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("正在处理视频...")

        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

    def _processing_worker(self) -> None:
        try:
            processor = VideoProcessor(self.video_path or "", self.config)
            processor.process(self._enqueue_preview, self.stop_event, self.pause_event)
        except Exception as exc:
            self._put_worker_message(("error", str(exc), None), force=True)

    def _enqueue_preview(self, frame, state: ProcessingState) -> None:
        if frame is None:
            self._put_worker_message(("done", state, None), force=True)
        else:
            self._put_worker_message(("frame", frame, state), force=False)

    def _put_worker_message(self, item: tuple[str, object, object | None], force: bool) -> None:
        try:
            self.result_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        if force:
            while True:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.result_queue.put_nowait(item)
                    return
                except queue.Full:
                    continue

        try:
            self.result_queue.get_nowait()
            self.result_queue.put_nowait(item)
        except queue.Empty:
            pass
        except queue.Full:
            pass

    def _toggle_pause(self) -> None:
        if not self._is_processing():
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="暂停")
            self.status_var.set("继续处理视频...")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="继续")
            self.status_var.set("已暂停。")

    def _stop_processing(self) -> None:
        if self._is_processing():
            self.stop_event.set()
            self.status_var.set("正在停止，请稍候...")

    def _poll_worker_queue(self) -> None:
        latest_frame = None
        latest_state: ProcessingState | None = None
        try:
            handled = 0
            while handled < 30:
                handled += 1
                kind, payload, extra = self.result_queue.get_nowait()
                if kind == "frame":
                    latest_frame = payload
                    latest_state = extra
                elif kind == "done":
                    if latest_frame is not None:
                        self._render_frame(latest_frame, overlay_setup=False)
                    if latest_state is not None:
                        self._update_state_labels(latest_state)
                    self._update_state_labels(payload)
                    self._finish_processing(payload)
                elif kind == "error":
                    self._finish_processing(None)
                    messagebox.showerror("处理失败", str(payload))
                    self.status_var.set(f"处理失败：{payload}")
        except queue.Empty:
            pass
        if latest_frame is not None:
            self._render_frame(latest_frame, overlay_setup=False)
        if latest_state is not None:
            self._update_state_labels(latest_state)
        self.after(80, self._poll_worker_queue)

    def _finish_processing(self, state: ProcessingState | None) -> None:
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED, text="暂停")
        self.stop_button.configure(state=tk.DISABLED)
        self.pause_event.clear()
        if state is None:
            return
        if state.stopped:
            self.status_var.set("处理已停止，已保存当前输出。")
        else:
            self.status_var.set("处理完成。")
        self.output_result_var.set(state.output_dir)

    def _open_output_dir(self) -> None:
        path = self.output_result_var.get() or self.output_dir_var.get()
        if not path:
            return
        target = Path(path)
        if not target.exists():
            messagebox.showwarning("目录不存在", f"找不到输出目录：{target}")
            return
        os.startfile(target)

    def _sync_config_from_vars(self, show_error: bool = False) -> bool:
        try:
            values = {}
            int_fields = {"history", "blur_size", "morph_kernel", "min_area", "max_area", "max_lost_frames", "track_history"}
            for attr, _label in self._param_specs():
                raw = self.param_vars[attr].get().strip()
                if attr in int_fields:
                    values[attr] = int(raw)
                else:
                    values[attr] = float(raw)
            params = DetectionParams(**values).normalized()
            manual_total = self._parse_optional_int(self.manual_total_var.get())
        except Exception as exc:
            if show_error:
                messagebox.showerror("参数错误", f"请检查检测参数和人工真值：{exc}")
            return False

        self.config.params = params
        self.config.output_dir = self.output_dir_var.get().strip() or str(Path.cwd() / "output")
        self.config.manual_total = manual_total
        self._load_params_to_vars(params)
        return True

    def _load_params_to_vars(self, params: DetectionParams) -> None:
        params = params.normalized()
        for attr, _label in self._param_specs():
            value = getattr(params, attr)
            if isinstance(value, float):
                self.param_vars[attr].set(str(round(value, 4)))
            else:
                self.param_vars[attr].set(str(value))

    def _parse_optional_int(self, value: str) -> int | None:
        value = value.strip()
        if not value:
            return None
        return max(0, int(value))

    def _render_setup_frame(self) -> None:
        if self.first_frame is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                max(20, self.canvas.winfo_width() // 2),
                max(20, self.canvas.winfo_height() // 2),
                text="请选择视频",
                fill="#dddddd",
                font=("Microsoft YaHei", 18),
            )
            return
        frame = draw_setup_overlay(self.first_frame, self.config.roi_points, self.config.line_points)
        self._render_frame(frame, overlay_setup=False)

    def _render_frame(self, frame, overlay_setup: bool = False) -> None:
        if overlay_setup:
            frame = draw_setup_overlay(frame, self.config.roi_points, self.config.line_points)

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        height, width = frame.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.preview_photo)
        self.display_scale = scale
        self.display_offset = (offset_x, offset_y)

    def _canvas_to_image_point(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        if self.first_frame is None:
            return None
        offset_x, offset_y = self.display_offset
        image_x = (canvas_x - offset_x) / self.display_scale
        image_y = (canvas_y - offset_y) / self.display_scale
        height, width = self.first_frame.shape[:2]
        if image_x < 0 or image_y < 0 or image_x >= width or image_y >= height:
            return None
        return int(image_x), int(image_y)

    def _update_state_labels(self, state: ProcessingState | None) -> None:
        if state is None:
            return
        self.progress_var.set(f"{state.frame_index} / {state.total_frames or '?'}")
        self.total_var.set(str(state.counts.get("total", 0)))
        self.active_tracks_var.set(str(state.active_tracks))
        self.frame_event_var.set(str(state.events_in_frame))
        self.processing_fps_var.set(f"{state.processing_fps:.1f}")
        self.elapsed_var.set(f"{state.elapsed_sec:.1f} 秒")
        if state.output_dir:
            self.output_result_var.set(state.output_dir)

    def _reset_counts(self) -> None:
        self.total_var.set("0")
        self.active_tracks_var.set("0")
        self.frame_event_var.set("0")
        self.processing_fps_var.set("0.0")
        self.elapsed_var.set("0.0 秒")

    def _is_processing(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()
