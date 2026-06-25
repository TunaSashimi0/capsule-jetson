from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class VideoSourceSpec:
    raw: str
    capture_source: int | str
    backend: int | None = None
    label: str = ""
    requires_gstreamer: bool = False


def cv2_has_gstreamer() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GStreamer:"):
            return "YES" in line.upper()
    return False


def jetson_csi_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def parse_video_source(source: str) -> VideoSourceSpec:
    value = source.strip()
    lowered = value.lower()

    if value.isdigit():
        return VideoSourceSpec(raw=value, capture_source=int(value), label=f"v4l2:{value}")

    if lowered in {"cam0", "csi0", "argus0"}:
        return _csi_source(value, 0)

    for prefix in ("cam:", "csi:", "argus:"):
        if lowered.startswith(prefix):
            sensor_id = int(lowered.removeprefix(prefix) or "0")
            return _csi_source(value, sensor_id)

    if lowered.startswith("gst:"):
        pipeline = value[4:].strip()
        return VideoSourceSpec(
            raw=value,
            capture_source=pipeline,
            backend=cv2.CAP_GSTREAMER,
            label="gstreamer",
            requires_gstreamer=True,
        )

    return VideoSourceSpec(raw=value, capture_source=value, label=value)


def open_video_capture(source: str) -> tuple[cv2.VideoCapture, VideoSourceSpec]:
    spec = parse_video_source(source)
    if spec.requires_gstreamer and not cv2_has_gstreamer():
        raise RuntimeError(
            "OpenCV was built without GStreamer support. CSI sources like "
            f"{source!r} need a Jetson/OpenCV build with GStreamer enabled."
        )

    if spec.backend is None:
        capture = cv2.VideoCapture(spec.capture_source)
    else:
        capture = cv2.VideoCapture(spec.capture_source, spec.backend)

    return capture, spec


def _csi_source(raw: str, sensor_id: int) -> VideoSourceSpec:
    return VideoSourceSpec(
        raw=raw,
        capture_source=jetson_csi_pipeline(sensor_id=sensor_id),
        backend=cv2.CAP_GSTREAMER,
        label=f"csi:{sensor_id}",
        requires_gstreamer=True,
    )
