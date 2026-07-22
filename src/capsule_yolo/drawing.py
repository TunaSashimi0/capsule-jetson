from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .counting import DEFECT_CLASS, GOOD_CLASS, CountSummary


DEFECT_COLOR = (0, 0, 255)
GOOD_COLOR = (40, 200, 80)
DEFAULT_COLOR = (255, 180, 40)


def draw_status_panel(
    frame: np.ndarray,
    summary: CountSummary,
    fps: float,
    conf: float,
    source_label: str,
) -> np.ndarray:
    output = frame.copy()
    panel_height = 218
    cv2.rectangle(output, (0, 0), (output.shape[1], panel_height), (20, 24, 31), -1)

    lines = [
        f"Total: {summary.capsule_count}",
        f"Good: {summary.good_count}",
        f"Defects: {summary.defect_count}",
        f"Avg W x H: {summary.avg_width_px:.1f} x {summary.avg_height_px:.1f} px",
        f"Avg Angle: {summary.avg_angle_deg:.1f} deg",
        f"FPS: {fps:.1f}",
        f"Confidence: {conf:.2f}",
        f"Source: {source_label}",
    ]
    for index, text in enumerate(lines):
        y = 28 + index * 25
        if index == 1:
            color = GOOD_COLOR
        elif index == 2:
            color = DEFECT_COLOR
        else:
            color = (255, 255, 255) if index == 0 else (210, 220, 230)
        cv2.putText(output, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    return output


def annotated_frame(result: Any, summary: CountSummary, fps: float, conf: float, source_label: str) -> np.ndarray:
    frame = result.orig_img.copy()
    line_width = max(2, round(min(frame.shape[:2]) / 500))

    for measurement in summary.measurements:
        color = _class_color(measurement.class_name)
        polygon = cv2.boxPoints(
            (
                (measurement.center_x_px, measurement.center_y_px),
                (measurement.width_px, measurement.height_px),
                measurement.angle_deg,
            )
        )
        points = np.rint(polygon).astype(np.int32)
        cv2.polylines(frame, [points], True, color, line_width, cv2.LINE_AA)
        _draw_label(
            frame,
            points,
            f"{measurement.class_name} {measurement.confidence:.2f}",
            color,
            line_width,
        )

    return draw_status_panel(frame, summary, fps=fps, conf=conf, source_label=source_label)


def _class_color(class_name: str) -> tuple[int, int, int]:
    if class_name == DEFECT_CLASS:
        return DEFECT_COLOR
    if class_name == GOOD_CLASS:
        return GOOD_COLOR
    return DEFAULT_COLOR


def _draw_label(
    frame: np.ndarray,
    points: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    font_scale = max(0.5, line_width / 4)
    font_thickness = max(1, line_width - 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    x = max(0, int(points[:, 0].min()))
    y = max(text_height + baseline + 4, int(points[:, 1].min()))
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 4),
        (min(frame.shape[1] - 1, x + text_width + 6), y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 3, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )
