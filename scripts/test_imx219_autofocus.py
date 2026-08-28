#!/usr/bin/env python3
"""Recognize and autofocus both Arducam IMX219-AF cameras on Orin Nano.

Arducam's B0181 actuator accepts a raw two-byte 10-bit focus position at I2C
address 0x0c. CAM A is sensor-id 0 / I2C bus 10; CAM C is sensor-id 1 / bus 9.
Run as root because the focus actuator is intentionally controlled through I2C.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


I2C_SLAVE = 0x0703
VCM_ADDRESS = 0x0C
CAMERAS = {0: 10, 1: 9}


@dataclass
class FocusResult:
    sensor_id: int
    i2c_bus: int
    coarse_best_position: int
    best_position: int
    best_score: float
    worst_score: float
    score_ratio: float
    lens_motion_visible: bool
    coarse_scores: dict[int, float]
    fine_scores: dict[int, float]


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def run(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def set_focus(bus: int, position: int) -> None:
    if not 0 <= position <= 1023:
        raise ValueError(f"focus position {position} is outside 0..1023")
    device = Path(f"/dev/i2c-{bus}")
    if not device.exists():
        fail(f"{device} is unavailable")

    # Arducam's published B0181 protocol shifts the 10-bit position left four
    # bits, then transmits the upper six bits followed by the lower nibble.
    value = (position << 4) & 0x3FF0
    payload = bytes(((value >> 8) & 0x3F, value & 0xF0))
    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, I2C_SLAVE, VCM_ADDRESS)
        written = os.write(fd, payload)
        if written != len(payload):
            fail(f"short I2C write on bus {bus}: {written}/{len(payload)} bytes")
    finally:
        os.close(fd)


class ArgusFrames:
    def __init__(self, sensor_id: int, width: int = 1280, height: int = 720) -> None:
        pipeline_text = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=GRAY8 ! "
            "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )
        self.pipeline = Gst.parse_launch(pipeline_text)
        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)
        result, state, _ = self.pipeline.get_state(10 * Gst.SECOND)
        if result == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
            self.close()
            fail(f"Argus sensor-id {sensor_id} failed to enter PLAYING state")

    def frame(self, timeout_seconds: int = 5) -> np.ndarray:
        sample = self.sink.emit("try-pull-sample", timeout_seconds * Gst.SECOND)
        if sample is None:
            message = self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            if message is not None:
                error, debug = message.parse_error()
                fail(f"GStreamer capture error: {error}; {debug}")
            fail("timed out waiting for an Argus frame")

        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            fail("could not map GStreamer frame buffer")
        try:
            raw = np.frombuffer(mapped.data, dtype=np.uint8)
            stride = raw.size // height
            if stride < width:
                fail(f"invalid frame stride {stride} for width {width}")
            return raw.reshape(height, stride)[:, :width].copy()
        finally:
            buffer.unmap(mapped)

    def settle_and_score(self, frames: int = 3) -> float:
        scores: list[float] = []
        for _ in range(frames):
            gray = self.frame()
            # Ignore the outer 5%, where lens shading and borders can dominate.
            y = max(1, gray.shape[0] // 20)
            x = max(1, gray.shape[1] // 20)
            roi = gray[y:-y, x:-x]
            scores.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
        return statistics.median(scores)

    def close(self) -> None:
        if getattr(self, "pipeline", None) is not None:
            self.pipeline.set_state(Gst.State.NULL)

    def __enter__(self) -> "ArgusFrames":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def autofocus(sensor_id: int, bus: int) -> FocusResult:
    print(f"\nAutofocus sensor-id {sensor_id} (expected I2C bus {bus})")
    coarse: dict[int, float] = {}
    fine: dict[int, float] = {}

    with ArgusFrames(sensor_id) as camera:
        # Warm up Argus auto-exposure before comparing sharpness.
        for _ in range(12):
            camera.frame()

        # The native sensor driver powers the camera module only while it is
        # streaming. Probe the VCM after Argus is running, as Arducam's example
        # does. Also check the other mux channel so an enumeration-order change
        # cannot make us report a false negative.
        actuator_errors: list[str] = []
        for candidate in (bus, next(value for value in CAMERAS.values() if value != bus)):
            try:
                set_focus(candidate, 512)
            except OSError as error:
                actuator_errors.append(f"bus {candidate}: {error}")
            else:
                bus = candidate
                print(f"  actuator acknowledged /dev/i2c-{bus} address 0x0c while streaming")
                break
        else:
            fail("focus actuator did not acknowledge while streaming (" + "; ".join(actuator_errors) + ")")

        for position in range(0, 1001, 50):
            set_focus(bus, position)
            time.sleep(0.08)
            score = camera.settle_and_score()
            coarse[position] = score
            print(f"  coarse {position:4d}: {score:10.3f}")

        coarse_best = max(coarse, key=coarse.get)
        fine_start = max(0, coarse_best - 60)
        fine_stop = min(1023, coarse_best + 60)
        for position in range(fine_start, fine_stop + 1, 10):
            set_focus(bus, position)
            time.sleep(0.08)
            score = camera.settle_and_score()
            fine[position] = score
            print(f"  fine   {position:4d}: {score:10.3f}")

        best_position = max(fine, key=fine.get)
        set_focus(bus, best_position)
        time.sleep(0.15)
        best_score = camera.settle_and_score(frames=5)

    all_scores = [*coarse.values(), *fine.values(), best_score]
    worst_score = min(all_scores)
    ratio = best_score / worst_score if worst_score > 0 else float("inf")
    # Successful commands prove the actuator answers. A >=15% image-metric
    # change additionally demonstrates visible lens motion in the current scene.
    result = FocusResult(
        sensor_id=sensor_id,
        i2c_bus=bus,
        coarse_best_position=coarse_best,
        best_position=best_position,
        best_score=best_score,
        worst_score=worst_score,
        score_ratio=ratio,
        lens_motion_visible=ratio >= 1.15,
        coarse_scores=coarse,
        fine_scores=fine,
    )
    print(
        f"  BEST position={best_position}, score={best_score:.3f}, "
        f"max/min={ratio:.2f}x, visible-motion={'yes' if result.lens_motion_visible else 'inconclusive'}"
    )
    return result


def recognition_report() -> None:
    devices = run(["v4l2-ctl", "--list-devices"])
    print("V4L2 devices:")
    print((devices.stdout or devices.stderr).strip())
    for sensor_id in CAMERAS:
        device = Path(f"/dev/video{sensor_id}")
        if not device.exists():
            fail(f"expected {device}; dual IMX219 overlay is not active or the camera did not probe")
        info = run(["v4l2-ctl", "-d", str(device), "--all"])
        if info.returncode:
            fail(f"cannot query {device}: {info.stderr.strip()}")
        if "imx219" not in info.stdout.lower():
            fail(f"{device} does not identify as IMX219")
        print(f"Recognized {device} as IMX219.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-id", type=int, choices=(0, 1), action="append", help="test only this sensor; repeat for both")
    parser.add_argument("--report", type=Path, default=Path("/tmp/imx219-autofocus-report.json"))
    parser.add_argument("--recognition-only", action="store_true")
    args = parser.parse_args()

    if os.geteuid() != 0 and not args.recognition_only:
        fail("autofocus I2C access requires root; run with sudo")

    Gst.init(None)
    recognition_report()
    if args.recognition_only:
        return 0

    sensor_ids = args.sensor_id or list(CAMERAS)
    results = [autofocus(sensor_id, CAMERAS[sensor_id]) for sensor_id in sensor_ids]
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": "NVIDIA Jetson Orin Nano Super",
        "results": [asdict(result) for result in results],
    }
    args.report.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nReport: {args.report}")
    if all(result.lens_motion_visible for result in results):
        print("PASS: both requested cameras streamed, accepted focus commands, and showed measurable focus changes.")
        return 0
    print("INCONCLUSIVE: streaming and focus I2C passed, but one scene lacked a strong measurable focus change.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
