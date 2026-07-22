from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.capsule_yolo.config import DEFAULT_MODEL_IMGSZ
from src.capsule_yolo.counting import CountSummary, summarize_result
from src.capsule_yolo.drawing import annotated_frame, draw_status_panel
from src.capsule_yolo.video_source import (
    DEFAULT_CAPTURE_FPS,
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    DEFAULT_ANALOG_GAIN,
    DEFAULT_DIGITAL_GAIN,
    DEFAULT_EXPOSURE_US,
    open_video_capture,
)


@dataclass
class CounterSettings:
    model: str
    source: str = "0"
    imgsz: int = DEFAULT_MODEL_IMGSZ
    conf: float = 0.25
    iou: float = 0.7
    device: str | None = None
    capture_width: int = DEFAULT_CAPTURE_WIDTH
    capture_height: int = DEFAULT_CAPTURE_HEIGHT
    capture_fps: int = DEFAULT_CAPTURE_FPS
    jpeg_quality: int = 95
    exposure_us: int = DEFAULT_EXPOSURE_US
    analog_gain: float = DEFAULT_ANALOG_GAIN
    digital_gain: float = DEFAULT_DIGITAL_GAIN
    half: bool = True


@dataclass
class CounterStats:
    capsule_count: int = 0
    good_count: int = 0
    defect_count: int = 0
    avg_width_px: float = 0.0
    avg_height_px: float = 0.0
    avg_angle_deg: float = 0.0
    measurements: list[dict[str, float | int | str]] = field(default_factory=list)
    fps: float = 0.0
    status: str = "idle"
    model: str = ""
    source: str = "0"
    conf: float = 0.25
    frame_time: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    highlight_clip_pct: float = 0.0
    mean_luma: float = 0.0
    precision: str = "FP32"


class VideoWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._stats = CounterStats()

    def start(self, settings: CounterSettings) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._stats = CounterStats(
                status="starting",
                model=settings.model,
                source=settings.source,
                conf=settings.conf,
            )
            self._thread = threading.Thread(target=self._run, args=(settings,), daemon=True)
            self._thread.start()

    def restart(self, settings: CounterSettings) -> None:
        self.stop()
        self.start(settings)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._thread = None
            self._stats.status = "stopped"

    def stats(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._stats)

    def frames(self) -> Generator[bytes, None, None]:
        while True:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg is None:
                jpeg = self._placeholder_jpeg("Waiting for video...")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)

    def _run(self, settings: CounterSettings) -> None:
        model_path = Path(settings.model)
        downloadable_models = {"yolo11n.pt", "yolo11n-obb.pt"}
        if not model_path.exists() and not settings.model.endswith(".pt"):
            self._set_placeholder(f"Model not found: {settings.model}", settings)
            return
        if not model_path.exists() and settings.model not in downloadable_models:
            self._set_placeholder(f"Train first: {settings.model}", settings)
            return

        try:
            model = YOLO(settings.model, task="obb")  #task obb
        except Exception as exc:
            self._set_placeholder(f"Model load failed: {exc}", settings)
            return

        use_half = settings.half and torch.cuda.is_available() and str(settings.device).lower() != "cpu"

        try:
            capture, source_spec = open_video_capture(
                settings.source,
                width=settings.capture_width,
                height=settings.capture_height,
                framerate=settings.capture_fps,
                exposure_us=settings.exposure_us,
                analog_gain=settings.analog_gain,
                digital_gain=settings.digital_gain,
            )
        except Exception as exc:
            self._set_placeholder(f"Source setup failed: {exc}", settings)
            return
        if not capture.isOpened():
            self._set_placeholder(f"Could not open source: {settings.source}", settings)
            return

        fps = 0.0
        last_time = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._set_placeholder(f"No frames from source: {settings.source}", settings)
                    break

                highlight_clip_pct = float(np.mean(np.max(frame, axis=2) >= 250) * 100.0)
                mean_luma = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                result = model.predict(
                    frame,
                    imgsz=settings.imgsz,
                    conf=settings.conf,
                    iou=settings.iou,
                    device=settings.device,
                    quantize=16 if use_half else 32,
                    verbose=False,
                )[0]
                summary = summarize_result(result)

                now = time.perf_counter()
                elapsed = max(now - last_time, 1e-6)
                fps = (0.85 * fps) + (0.15 * (1.0 / elapsed)) if fps else 1.0 / elapsed
                last_time = now

                frame_out = annotated_frame(
                    result,
                    summary,
                    fps=fps,
                    conf=settings.conf,
                    source_label=source_spec.label or settings.source,
                )
                frame_height, frame_width = frame_out.shape[:2]
                jpeg = self._encode_jpeg(frame_out, quality=settings.jpeg_quality)
                with self._lock:
                    self._latest_jpeg = jpeg
                    self._stats = CounterStats(
                        capsule_count=summary.capsule_count,
                        good_count=summary.good_count,
                        defect_count=summary.defect_count,
                        avg_width_px=summary.avg_width_px,
                        avg_height_px=summary.avg_height_px,
                        avg_angle_deg=summary.avg_angle_deg,
                        measurements=summary.measurements_as_dicts(limit=20),
                        fps=fps,
                        status="running",
                        model=settings.model,
                        source=settings.source,
                        conf=settings.conf,
                        frame_time=now,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        highlight_clip_pct=highlight_clip_pct,
                        mean_luma=mean_luma,
                        precision="FP16" if use_half else "FP32",
                    )
        except Exception as exc:
            self._set_placeholder(f"Inference failed: {exc}", settings)
        finally:
            capture.release()

    def _set_placeholder(self, message: str, settings: CounterSettings) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        summary = CountSummary(total=0, by_class={"capsule_good": 0, "capsule_defect": 0})
        frame = draw_status_panel(frame, summary, fps=0.0, conf=settings.conf, source_label=settings.source)
        cv2.putText(frame, message[:90], (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
        with self._lock:
            self._latest_jpeg = self._encode_jpeg(frame)
            self._stats = CounterStats(
                measurements=[],
                status=message,
                model=settings.model,
                source=settings.source,
                conf=settings.conf,
            )

    def _placeholder_jpeg(self, message: str) -> bytes:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, message, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        return self._encode_jpeg(frame)

    @staticmethod
    def _encode_jpeg(frame: np.ndarray, quality: int = 95) -> bytes:
        quality = max(1, min(100, quality))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return encoded.tobytes()
