from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from ultralytics import YOLO

from src.capsule_yolo.config import parse_source
from src.capsule_yolo.counting import CountSummary, summarize_result
from src.capsule_yolo.drawing import annotated_frame, draw_status_panel


@dataclass
class CounterSettings:
    model: str
    source: str = "0"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str | None = None


@dataclass
class CounterStats:
    capsule_count: int = 0
    fps: float = 0.0
    status: str = "idle"
    model: str = ""
    source: str = "0"
    conf: float = 0.25
    frame_time: float = 0.0


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

    def stats(self) -> dict[str, float | int | str]:
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
        if not model_path.exists() and not settings.model.endswith(".pt"):
            self._set_placeholder(f"Model not found: {settings.model}", settings)
            return
        if not model_path.exists() and settings.model != "yolo11n.pt":
            self._set_placeholder(f"Train first: {settings.model}", settings)
            return

        try:
            model = YOLO(settings.model)
        except Exception as exc:
            self._set_placeholder(f"Model load failed: {exc}", settings)
            return

        capture = cv2.VideoCapture(parse_source(settings.source))
        if not capture.isOpened():
            self._set_placeholder(f"Could not open source: {settings.source}", settings)
            return

        fps = 0.0
        last_time = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                result = model.predict(
                    frame,
                    imgsz=settings.imgsz,
                    conf=settings.conf,
                    iou=settings.iou,
                    device=settings.device,
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
                    source_label=settings.source,
                )
                jpeg = self._encode_jpeg(frame_out)
                with self._lock:
                    self._latest_jpeg = jpeg
                    self._stats = CounterStats(
                        capsule_count=summary.capsule_count,
                        fps=fps,
                        status="running",
                        model=settings.model,
                        source=settings.source,
                        conf=settings.conf,
                        frame_time=now,
                    )
        except Exception as exc:
            self._set_placeholder(f"Inference failed: {exc}", settings)
        finally:
            capture.release()

    def _set_placeholder(self, message: str, settings: CounterSettings) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        summary = CountSummary(total=0, by_class={"capsule": 0})
        frame = draw_status_panel(frame, summary, fps=0.0, conf=settings.conf, source_label=settings.source)
        cv2.putText(frame, message[:90], (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
        with self._lock:
            self._latest_jpeg = self._encode_jpeg(frame)
            self._stats = CounterStats(
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
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return encoded.tobytes()
