from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .counting import CountSummary


def draw_status_panel(
    frame: np.ndarray,
    summary: CountSummary,
    fps: float,
    conf: float,
    source_label: str,
) -> np.ndarray:
    output = frame.copy()
    panel_height = 168
    cv2.rectangle(output, (0, 0), (output.shape[1], panel_height), (20, 24, 31), -1)

    lines = [
        f"Capsules: {summary.capsule_count}",
        f"Avg W x H: {summary.avg_width_px:.1f} x {summary.avg_height_px:.1f} px",
        f"Avg Angle: {summary.avg_angle_deg:.1f} deg",
        f"FPS: {fps:.1f}",
        f"Confidence: {conf:.2f}",
        f"Source: {source_label}",
    ]
    for index, text in enumerate(lines):
        y = 28 + index * 25
        color = (255, 255, 255) if index == 0 else (210, 220, 230)
        cv2.putText(output, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    return output


def annotated_frame(result: Any, summary: CountSummary, fps: float, conf: float, source_label: str) -> np.ndarray:
    frame = result.plot()
    return draw_status_panel(frame, summary, fps=fps, conf=conf, source_label=source_label)
