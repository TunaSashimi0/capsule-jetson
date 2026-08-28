from __future__ import annotations

import fcntl
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


I2C_SLAVE = 0x0703
VCM_ADDRESS = 0x0C
SENSOR_I2C_BUSES = {0: 10, 1: 9}


@dataclass(frozen=True)
class FocusScanResult:
    i2c_bus: int
    best_position: int
    best_score: float
    worst_score: float
    score_ratio: float


def focus_bus_for_sensor(sensor_id: int) -> int:
    try:
        return SENSOR_I2C_BUSES[sensor_id]
    except KeyError as exc:
        raise ValueError(f"No Arducam focus-bus mapping for sensor-id {sensor_id}") from exc


def set_focus(bus: int, position: int) -> None:
    """Set a B0181 IMX219-AF lens to a 10-bit focus position."""
    if not 0 <= position <= 1023:
        raise ValueError(f"focus position {position} is outside 0..1023")
    device = Path(f"/dev/i2c-{bus}")
    if not device.exists():
        raise FileNotFoundError(f"{device} is unavailable")

    value = (position << 4) & 0x3FF0
    payload = bytes(((value >> 8) & 0x3F, value & 0xF0))
    fd = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(fd, I2C_SLAVE, VCM_ADDRESS)
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"short I2C write on bus {bus}: {written}/{len(payload)} bytes")
    finally:
        os.close(fd)


def sharpness_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    y_margin = max(1, gray.shape[0] // 20)
    x_margin = max(1, gray.shape[1] // 20)
    roi = gray[y_margin:-y_margin, x_margin:-x_margin]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def autofocus_capture(
    capture: cv2.VideoCapture,
    bus: int,
    *,
    stop_event: threading.Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> FocusScanResult:
    """Autofocus an already-streaming B0181 camera without reducing frame size."""

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def check_stopped() -> None:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("autofocus stopped")

    def read_score(frame_count: int = 2) -> float:
        scores: list[float] = []
        for _ in range(frame_count):
            check_stopped()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera stopped delivering frames during autofocus")
            scores.append(sharpness_score(frame))
        return statistics.median(scores)

    # The module rails, including the VCM, are powered only while Argus is
    # streaming. Warm exposure first, then verify the actuator at a safe point.
    for _ in range(10):
        check_stopped()
        ok, _ = capture.read()
        if not ok:
            raise RuntimeError("camera stopped delivering autofocus warm-up frames")
    set_focus(bus, 512)

    coarse_scores: dict[int, float] = {}
    for position in range(0, 1001, 50):
        check_stopped()
        set_focus(bus, position)
        time.sleep(0.06)
        coarse_scores[position] = read_score()
        report(f"coarse scan {position}/1000")

    coarse_best = max(coarse_scores, key=coarse_scores.get)
    fine_scores: dict[int, float] = {}
    fine_start = max(0, coarse_best - 60)
    fine_stop = min(1023, coarse_best + 60)
    for position in range(fine_start, fine_stop + 1, 10):
        check_stopped()
        set_focus(bus, position)
        time.sleep(0.06)
        fine_scores[position] = read_score()
        report(f"fine scan {position}")

    best_position = max(fine_scores, key=fine_scores.get)
    set_focus(bus, best_position)
    time.sleep(0.12)
    best_score = read_score(frame_count=4)
    scores = [*coarse_scores.values(), *fine_scores.values(), best_score]
    worst_score = min(scores)
    score_ratio = best_score / worst_score if worst_score > 0 else float("inf")
    report(f"focused at {best_position} ({score_ratio:.1f}x sharpness range)")
    return FocusScanResult(
        i2c_bus=bus,
        best_position=best_position,
        best_score=best_score,
        worst_score=worst_score,
        score_ratio=score_ratio,
    )
