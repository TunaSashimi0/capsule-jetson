from __future__ import annotations

import asyncio
import copy
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
from ultralytics import YOLO

from src.capsule_yolo.arducam_focus import autofocus_capture, focus_bus_for_sensor, set_focus
from src.capsule_yolo.counting import CountSummary, summarize_result
from src.capsule_yolo.drawing import draw_status_panel, lightweight_preview
from src.capsule_yolo.video_source import VideoSourceSpec, open_video_capture


@dataclass
class CounterSettings:
    model: str
    source: str = "csi:0"
    secondary_source: str | None = "csi:1"
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.7
    device: str | None = None
    capture_width: int = 3280
    capture_height: int = 2464
    capture_fps: int = 21
    preview_width: int = 1280
    preview_fps: float = 2.0
    preview_jpeg_quality: int = 84
    autofocus: bool = True

    def sources(self) -> tuple[str, ...]:
        values = [self.source]
        if self.secondary_source and self.secondary_source.strip():
            values.append(self.secondary_source.strip())
        return tuple(values)


@dataclass
class CameraStats:
    camera_index: int
    camera_label: str
    source: str
    capsule_count: int = 0
    good_count: int = 0
    defect_count: int = 0
    avg_width_px: float = 0.0
    avg_height_px: float = 0.0
    avg_angle_deg: float = 0.0
    measurements: list[dict[str, float | int | str]] = field(default_factory=list)
    fps: float = 0.0
    status: str = "idle"
    frame_time: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    focus_status: str = "pending"
    focus_position: int | None = None
    focus_score_ratio: float | None = None
    inference_count: int = 0


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
    source: str = ""
    conf: float = 0.25
    frame_time: float = 0.0
    cameras: list[CameraStats] = field(default_factory=list)
    inference_width: int = 1280
    capture_width: int = 3280
    capture_height: int = 2464
    preview_width: int = 1280
    preview_fps: float = 2.0
    inference_count: int = 0


@dataclass(frozen=True)
class PreviewPacket:
    camera_index: int
    frame: np.ndarray
    summary: CountSummary
    fps: float
    source_label: str


class VideoWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None
        self._latest_jpegs: dict[int, bytes] = {}
        self._preview_sequences: dict[int, int] = {}
        self._preview_clients: dict[int, int] = {}
        self._pending_previews: dict[int, PreviewPacket] = {}
        self._preview_condition = threading.Condition(self._lock)
        self._placeholder_cache: dict[tuple[str, int, int], bytes] = {}
        self._stats = CounterStats()

    def start(self, settings: CounterSettings) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._latest_jpegs.clear()
            self._pending_previews.clear()
            camera_stats = self._initial_camera_stats(settings, status="starting")
            self._stats = self._aggregate_stats(camera_stats, settings, status="starting")
            self._preview_thread = threading.Thread(
                target=self._preview_loop,
                args=(settings,),
                daemon=True,
            )
            self._thread = threading.Thread(target=self._run, args=(settings,), daemon=True)
            self._preview_thread.start()
            self._thread.start()

    def restart(self, settings: CounterSettings) -> None:
        self.stop()
        self.start(settings)

    def stop(self) -> None:
        self._stop_event.set()
        with self._preview_condition:
            self._preview_condition.notify_all()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=15.0)
        preview_thread = self._preview_thread
        if preview_thread and preview_thread.is_alive():
            preview_thread.join(timeout=3.0)
        with self._lock:
            if not (self._thread and self._thread.is_alive()):
                self._thread = None
            if not (self._preview_thread and self._preview_thread.is_alive()):
                self._preview_thread = None
            self._pending_previews.clear()
            self._stats.status = "stopped"
            for camera in self._stats.cameras:
                camera.status = "stopped"

    def stats(self) -> dict[str, object]:
        with self._lock:
            payload = asdict(self._stats)
            payload["preview_clients"] = sum(self._preview_clients.values())
            payload["preview_sequences"] = [
                self._preview_sequences.get(index, 0)
                for index in range(len(self._stats.cameras))
            ]
            return payload

    async def frames(self, camera_index: int = 0) -> AsyncGenerator[bytes, None]:
        with self._preview_condition:
            self._preview_clients[camera_index] = self._preview_clients.get(camera_index, 0) + 1
            self._preview_condition.notify_all()

        last_sequence = -1
        last_yield = 0.0
        try:
            while True:
                with self._lock:
                    jpeg = self._latest_jpegs.get(camera_index)
                    sequence = self._preview_sequences.get(camera_index, 0)
                    camera_count = len(self._stats.cameras)
                    preview_width = self._stats.preview_width
                if camera_index >= camera_count:
                    jpeg = self._placeholder_jpeg(
                        f"Camera {camera_index} is not configured",
                        preview_width,
                    )
                elif jpeg is None:
                    jpeg = self._placeholder_jpeg(
                        f"Waiting for camera {camera_index}...",
                        preview_width,
                    )

                now = time.monotonic()
                if sequence != last_sequence or now - last_yield >= 2.0:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    last_sequence = sequence
                    last_yield = now
                await asyncio.sleep(0.05)
        finally:
            with self._preview_condition:
                remaining = max(0, self._preview_clients.get(camera_index, 1) - 1)
                if remaining:
                    self._preview_clients[camera_index] = remaining
                else:
                    self._preview_clients.pop(camera_index, None)
                    self._pending_previews.pop(camera_index, None)
                self._preview_condition.notify_all()

    def _run(self, settings: CounterSettings) -> None:
        sources = settings.sources()
        camera_states = self._initial_camera_stats(settings, status="loading model")
        self._publish(camera_states, settings, status="loading model")

        model_path = Path(settings.model)
        downloadable_models = {"yolo11n.pt", "yolo11n-obb.pt"}
        if not model_path.exists() and not settings.model.endswith(".pt"):
            self._set_placeholders(f"Model not found: {settings.model}", settings)
            return
        if not model_path.exists() and settings.model not in downloadable_models:
            self._set_placeholders(f"Train first: {settings.model}", settings)
            return

        try:
            model = YOLO(settings.model, task="obb")
        except Exception as exc:
            self._set_placeholders(f"Model load failed: {exc}", settings)
            return

        captures: list[cv2.VideoCapture] = []
        specs: list[VideoSourceSpec] = []
        try:
            for index, source in enumerate(sources):
                camera_states[index].status = "opening full-resolution stream"
                self._publish(camera_states, settings, status="opening cameras")
                capture, source_spec = open_video_capture(
                    source,
                    width=settings.capture_width,
                    height=settings.capture_height,
                    framerate=settings.capture_fps,
                )
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"could not open source {source!r}")
                captures.append(capture)
                specs.append(source_spec)

            self._autofocus(captures, specs, camera_states, settings)
            if self._stop_event.is_set():
                return

            per_camera_fps = [0.0 for _ in captures]
            last_times = [time.perf_counter() for _ in captures]
            active = [True for _ in captures]

            while not self._stop_event.is_set() and any(active):
                for index, (capture, spec) in enumerate(zip(captures, specs)):
                    if self._stop_event.is_set() or not active[index]:
                        continue
                    ok, frame = capture.read()
                    if not ok:
                        camera_states[index].status = f"reconnecting {sources[index]}"
                        self._publish(camera_states, settings, status="camera reconnecting")
                        replacement = self._reconnect_capture(
                            source=sources[index],
                            old_capture=capture,
                            settings=settings,
                            focus_position=camera_states[index].focus_position,
                            sensor_id=spec.sensor_id,
                        )
                        if replacement is None:
                            active[index] = False
                            camera_states[index].status = f"no frames from {sources[index]}"
                            self._publish(camera_states, settings, status="camera error")
                        else:
                            captures[index], specs[index] = replacement
                            last_times[index] = time.perf_counter()
                        continue

                    # Keep the native frame intact through inference. Only the
                    # annotated browser preview is resized later.
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
                    elapsed = max(now - last_times[index], 1e-6)
                    instant_fps = 1.0 / elapsed
                    per_camera_fps[index] = (
                        (0.85 * per_camera_fps[index]) + (0.15 * instant_fps)
                        if per_camera_fps[index]
                        else instant_fps
                    )
                    last_times[index] = now

                    camera_states[index] = self._camera_stats_from_summary(
                        previous=camera_states[index],
                        summary=summary,
                        frame=frame,
                        fps=per_camera_fps[index],
                        now=now,
                    )
                    self._publish(camera_states, settings, status="running")
                    self._offer_preview(
                        PreviewPacket(
                            camera_index=index,
                            frame=frame,
                            summary=summary,
                            fps=per_camera_fps[index],
                            source_label=spec.label or sources[index],
                        )
                    )
        except InterruptedError:
            pass
        except Exception as exc:
            self._set_placeholders(f"Inference failed: {exc}", settings, camera_states)
        finally:
            for capture in captures:
                capture.release()

    def _reconnect_capture(
        self,
        *,
        source: str,
        old_capture: cv2.VideoCapture,
        settings: CounterSettings,
        focus_position: int | None,
        sensor_id: int | None,
    ) -> tuple[cv2.VideoCapture, VideoSourceSpec] | None:
        """Bounded recovery for an Argus stream that stopped producing frames."""
        old_capture.release()
        for _ in range(3):
            if self._stop_event.wait(0.25):
                return None
            replacement: cv2.VideoCapture | None = None
            try:
                replacement, spec = open_video_capture(
                    source,
                    width=settings.capture_width,
                    height=settings.capture_height,
                    framerate=settings.capture_fps,
                )
                if not replacement.isOpened():
                    replacement.release()
                    continue
                ok, _ = replacement.read()
                if not ok:
                    replacement.release()
                    continue
                if settings.autofocus and focus_position is not None and sensor_id is not None:
                    set_focus(focus_bus_for_sensor(sensor_id), focus_position)
                return replacement, spec
            except Exception as exc:
                print(f"Camera reconnect failed for {source}: {exc}", flush=True)
                if replacement is not None:
                    replacement.release()
        return None

    def _offer_preview(self, packet: PreviewPacket) -> None:
        """Keep only the newest frame; inference never waits for preview work."""
        with self._preview_condition:
            if self._preview_clients.get(packet.camera_index, 0) <= 0:
                return
            self._pending_previews[packet.camera_index] = packet
            self._preview_condition.notify_all()

    def _preview_loop(self, settings: CounterSettings) -> None:
        interval = 1.0 / max(0.1, settings.preview_fps)
        next_due: dict[int, float] = {}
        while not self._stop_event.is_set():
            packets: list[PreviewPacket] = []
            with self._preview_condition:
                now = time.monotonic()
                for index in list(self._pending_previews):
                    if self._preview_clients.get(index, 0) <= 0:
                        self._pending_previews.pop(index, None)
                    elif now >= next_due.get(index, 0.0):
                        packets.append(self._pending_previews.pop(index))
                        next_due[index] = now + interval
                if not packets:
                    self._preview_condition.wait(timeout=0.1)
                    continue

            for packet in packets:
                if self._stop_event.is_set():
                    return
                try:
                    frame_out = lightweight_preview(
                        packet.frame,
                        packet.summary,
                        max_width=settings.preview_width,
                        fps=packet.fps,
                        source_label=packet.source_label,
                    )
                    jpeg = self._encode_preview(
                        frame_out,
                        0,
                        quality=settings.preview_jpeg_quality,
                    )
                except Exception as exc:
                    print(
                        f"Preview render failed for camera {packet.camera_index}: {exc}",
                        flush=True,
                    )
                    continue
                with self._lock:
                    if self._preview_clients.get(packet.camera_index, 0) > 0:
                        self._latest_jpegs[packet.camera_index] = jpeg
                        self._preview_sequences[packet.camera_index] = (
                            self._preview_sequences.get(packet.camera_index, 0) + 1
                        )

    def _autofocus(
        self,
        captures: list[cv2.VideoCapture],
        specs: list[VideoSourceSpec],
        camera_states: list[CameraStats],
        settings: CounterSettings,
    ) -> None:
        for index, (capture, spec) in enumerate(zip(captures, specs)):
            if not settings.autofocus:
                camera_states[index].focus_status = "disabled"
                camera_states[index].status = "ready"
                continue
            if spec.sensor_id is None:
                camera_states[index].focus_status = "not available for this source"
                camera_states[index].status = "ready"
                continue

            bus = focus_bus_for_sensor(spec.sensor_id)

            def progress(message: str, camera_index: int = index) -> None:
                camera_states[camera_index].status = f"autofocus: {message}"
                camera_states[camera_index].focus_status = message
                self._publish(camera_states, settings, status="autofocusing")

            camera_states[index].status = "autofocus: warming up"
            camera_states[index].focus_status = "warming up"
            self._publish(camera_states, settings, status="autofocusing")
            try:
                result = autofocus_capture(
                    capture,
                    bus,
                    stop_event=self._stop_event,
                    progress=progress,
                )
            except InterruptedError:
                raise
            except Exception as exc:
                camera_states[index].focus_status = f"failed: {exc}"
                camera_states[index].status = "running without autofocus"
            else:
                camera_states[index].focus_status = "focused"
                camera_states[index].focus_position = result.best_position
                camera_states[index].focus_score_ratio = result.score_ratio
                camera_states[index].status = "ready"
            self._publish(camera_states, settings, status="autofocusing")

    @staticmethod
    def _initial_camera_stats(settings: CounterSettings, status: str) -> list[CameraStats]:
        return [
            CameraStats(
                camera_index=index,
                camera_label=f"Camera {'A' if index == 0 else 'C' if index == 1 else index}",
                source=source,
                status=status,
                focus_status="pending" if settings.autofocus else "disabled",
            )
            for index, source in enumerate(settings.sources())
        ]

    @staticmethod
    def _camera_stats_from_summary(
        *,
        previous: CameraStats,
        summary: CountSummary,
        frame: np.ndarray,
        fps: float,
        now: float,
    ) -> CameraStats:
        return CameraStats(
            camera_index=previous.camera_index,
            camera_label=previous.camera_label,
            source=previous.source,
            capsule_count=summary.capsule_count,
            good_count=summary.good_count,
            defect_count=summary.defect_count,
            avg_width_px=summary.avg_width_px,
            avg_height_px=summary.avg_height_px,
            avg_angle_deg=summary.avg_angle_deg,
            measurements=summary.measurements_as_dicts(limit=20),
            fps=fps,
            status="running",
            frame_time=now,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            focus_status=previous.focus_status,
            focus_position=previous.focus_position,
            focus_score_ratio=previous.focus_score_ratio,
            inference_count=previous.inference_count + 1,
        )

    def _publish(
        self,
        cameras: list[CameraStats],
        settings: CounterSettings,
        *,
        status: str,
    ) -> None:
        stats = self._aggregate_stats(cameras, settings, status=status)
        with self._lock:
            self._stats = stats

    @staticmethod
    def _aggregate_stats(
        cameras: list[CameraStats],
        settings: CounterSettings,
        *,
        status: str,
    ) -> CounterStats:
        measurements: list[dict[str, float | int | str]] = []
        for camera in cameras:
            for item in camera.measurements:
                measurements.append(
                    {
                        **item,
                        "camera_index": camera.camera_index,
                        "camera_label": camera.camera_label,
                    }
                )

        def average(key: str) -> float:
            values = [float(item[key]) for item in measurements if key in item]
            return sum(values) / len(values) if values else 0.0

        return CounterStats(
            capsule_count=sum(item.capsule_count for item in cameras),
            good_count=sum(item.good_count for item in cameras),
            defect_count=sum(item.defect_count for item in cameras),
            avg_width_px=average("width_px"),
            avg_height_px=average("height_px"),
            avg_angle_deg=average("angle_deg"),
            measurements=measurements[:40],
            fps=sum(item.fps for item in cameras),
            status=status,
            model=settings.model,
            source=", ".join(settings.sources()),
            conf=settings.conf,
            frame_time=max((item.frame_time for item in cameras), default=0.0),
            cameras=copy.deepcopy(cameras),
            inference_width=settings.imgsz,
            capture_width=settings.capture_width,
            capture_height=settings.capture_height,
            preview_width=settings.preview_width,
            preview_fps=settings.preview_fps,
            inference_count=sum(item.inference_count for item in cameras),
        )

    def _set_placeholders(
        self,
        message: str,
        settings: CounterSettings,
        cameras: list[CameraStats] | None = None,
    ) -> None:
        camera_states = cameras or self._initial_camera_stats(settings, status=message)
        for index, camera in enumerate(camera_states):
            camera.status = message
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            summary = CountSummary(total=0, by_class={"capsule_good": 0, "capsule_defect": 0})
            frame = draw_status_panel(
                frame,
                summary,
                fps=0.0,
                conf=settings.conf,
                source_label=camera.source,
            )
            cv2.putText(
                frame,
                message[:90],
                (40, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            with self._lock:
                self._latest_jpegs[index] = self._encode_preview(
                    frame,
                    settings.preview_width,
                    quality=settings.preview_jpeg_quality,
                )
                self._preview_sequences[index] = self._preview_sequences.get(index, 0) + 1
        self._publish(camera_states, settings, status=message)

    def _placeholder_jpeg(self, message: str, width: int = 640) -> bytes:
        cache_key = (message, width, 70)
        with self._lock:
            cached = self._placeholder_cache.get(cache_key)
        if cached is not None:
            return cached
        height = max(240, round(width * 0.75))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            message,
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        jpeg = self._encode_preview(frame, width, quality=70)
        with self._lock:
            self._placeholder_cache[cache_key] = jpeg
        return jpeg

    @staticmethod
    def _encode_preview(frame: np.ndarray, max_width: int, *, quality: int = 70) -> bytes:
        preview = frame
        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            preview = cv2.resize(
                frame,
                (max_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        quality = min(95, max(30, quality))
        ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return encoded.tobytes()
