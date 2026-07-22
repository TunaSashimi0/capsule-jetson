from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import cv2


DEFAULT_CAPTURE_WIDTH = 1920
DEFAULT_CAPTURE_HEIGHT = 1080
DEFAULT_CAPTURE_FPS = 30
DEFAULT_EXPOSURE_US = 8000
DEFAULT_ANALOG_GAIN = 1.0
DEFAULT_DIGITAL_GAIN = 1.0


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
    exposure_us: int = DEFAULT_EXPOSURE_US,
    analog_gain: float = DEFAULT_ANALOG_GAIN,
    digital_gain: float = DEFAULT_DIGITAL_GAIN,
) -> str:
    camera_properties: list[str] = []
    if exposure_us > 0:
        exposure_ns = exposure_us * 1000
        camera_properties.extend(
            [f'exposuretimerange="{exposure_ns} {exposure_ns}"', "aelock=true"]
        )
    if analog_gain > 0:
        camera_properties.append(f'gainrange="{analog_gain:g} {analog_gain:g}"')
    if digital_gain > 0:
        camera_properties.append(f'ispdigitalgainrange="{digital_gain:g} {digital_gain:g}"')
    properties = " ".join(camera_properties)
    if properties:
        properties = f" {properties}"

    return (
        f"nvarguscamerasrc sensor-id={sensor_id}{properties} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def parse_video_source(
    source: str,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    framerate: int = DEFAULT_CAPTURE_FPS,
    exposure_us: int = DEFAULT_EXPOSURE_US,
    analog_gain: float = DEFAULT_ANALOG_GAIN,
    digital_gain: float = DEFAULT_DIGITAL_GAIN,
) -> VideoSourceSpec:
    value = source.strip()
    lowered = value.lower()

    if value.isdigit():
        return VideoSourceSpec(raw=value, capture_source=int(value), label=f"v4l2:{value}")

    if lowered in {"cam0", "csi0", "argus0"}:
        return _csi_source(
            value, 0, width, height, framerate, exposure_us, analog_gain, digital_gain
        )

    for prefix in ("cam:", "csi:", "argus:"):
        if lowered.startswith(prefix):
            sensor_id = int(lowered.removeprefix(prefix) or "0")
            return _csi_source(
                value,
                sensor_id,
                width,
                height,
                framerate,
                exposure_us,
                analog_gain,
                digital_gain,
            )

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


def open_video_capture(
    source: str,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    framerate: int = DEFAULT_CAPTURE_FPS,
    exposure_us: int = DEFAULT_EXPOSURE_US,
    analog_gain: float = DEFAULT_ANALOG_GAIN,
    digital_gain: float = DEFAULT_DIGITAL_GAIN,
) -> tuple[cv2.VideoCapture, VideoSourceSpec]:
    spec = parse_video_source(
        source,
        width=width,
        height=height,
        framerate=framerate,
        exposure_us=exposure_us,
        analog_gain=analog_gain,
        digital_gain=digital_gain,
    )
    if spec.requires_gstreamer and not cv2_has_gstreamer():
        raise RuntimeError(
            "OpenCV was built without GStreamer support. CSI sources like "
            f"{source!r} need a Jetson/OpenCV build with GStreamer enabled."
        )

    if spec.backend is None:
        capture = cv2.VideoCapture(spec.capture_source)
    else:
        capture = cv2.VideoCapture(spec.capture_source, spec.backend)

    if isinstance(spec.capture_source, int):
        # Camera backends may otherwise negotiate a low default such as 640x480.
        # These are requests; unsupported cameras retain their closest mode.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, framerate)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _configure_standard_camera(capture, exposure_us=exposure_us, analog_gain=analog_gain)

    return capture, spec


def _csi_source(
    raw: str,
    sensor_id: int,
    width: int,
    height: int,
    framerate: int,
    exposure_us: int,
    analog_gain: float,
    digital_gain: float,
) -> VideoSourceSpec:
    return VideoSourceSpec(
        raw=raw,
        capture_source=jetson_csi_pipeline(
            sensor_id=sensor_id,
            width=width,
            height=height,
            framerate=framerate,
            exposure_us=exposure_us,
            analog_gain=analog_gain,
            digital_gain=digital_gain,
        ),
        backend=cv2.CAP_GSTREAMER,
        label=f"csi:{sensor_id}",
        requires_gstreamer=True,
    )


def _configure_standard_camera(
    capture: cv2.VideoCapture,
    exposure_us: int,
    analog_gain: float,
) -> None:
    if exposure_us > 0:
        if sys.platform == "win32":
            # DirectShow commonly uses 0.25 for manual exposure and a base-2
            # seconds scale for CAP_PROP_EXPOSURE (for example, -9 is ~2 ms).
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            exposure_value = round(math.log2(exposure_us / 1_000_000))
        else:
            # V4L2 exposure_absolute is commonly expressed in 100 us units.
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            exposure_value = exposure_us / 100
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    else:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if sys.platform == "win32" else 3.0)

    if analog_gain >= 0:
        capture.set(cv2.CAP_PROP_GAIN, analog_gain)
