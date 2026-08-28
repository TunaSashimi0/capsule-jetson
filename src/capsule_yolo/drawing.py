from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .counting import CountSummary


DEFECT_CLASS = "capsule_defect"
GOOD_CLASS = "capsule_good"
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
    detections = getattr(result, "obb", None)
    polygons = getattr(detections, "xyxyxyxy", None) if detections is not None else None
    class_ids = getattr(detections, "cls", None) if detections is not None else None
    confidences = getattr(detections, "conf", None) if detections is not None else None
    if polygons is None or class_ids is None:
        frame = result.plot()
        return draw_status_panel(frame, summary, fps=fps, conf=conf, source_label=source_label)

    names = getattr(result, "names", {}) or {}
    polygons = _to_numpy(polygons)
    class_ids = _to_numpy(class_ids).reshape(-1)
    confidences = _to_numpy(confidences).reshape(-1) if confidences is not None else np.zeros(len(class_ids))
    line_width = max(2, round(min(frame.shape[:2]) / 500))

    for index, polygon in enumerate(polygons):
        class_id = int(class_ids[index])
        class_name = names.get(class_id, names.get(str(class_id), str(class_id)))
        color = _class_color(class_name)
        points = np.rint(polygon).astype(np.int32).reshape(-1, 2)
        cv2.polylines(frame, [points], True, color, line_width, cv2.LINE_AA)
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        _draw_label(frame, points, f"{class_name} {confidence:.2f}", color, line_width)

    return draw_status_panel(frame, summary, fps=fps, conf=conf, source_label=source_label)


def lightweight_preview(
    frame: np.ndarray,
    summary: CountSummary,
    *,
    max_width: int,
    fps: float,
    source_label: str,
) -> np.ndarray:
    """Render a small annotated preview without copying/drawing the native frame."""
    scale = 1.0
    if max_width > 0 and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        output = cv2.resize(
            frame,
            (max_width, max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        output = frame.copy()

    line_width = max(1, round(min(output.shape[:2]) / 400))
    for item in summary.measurements:
        color = _class_color(item.class_name)
        rectangle = (
            (item.center_x_px * scale, item.center_y_px * scale),
            (item.width_px * scale, item.height_px * scale),
            item.angle_deg,
        )
        points = np.rint(cv2.boxPoints(rectangle)).astype(np.int32)
        cv2.polylines(output, [points], True, color, line_width, cv2.LINE_AA)
        _draw_label(
            output,
            points,
            f"{item.class_name} {item.confidence:.2f}",
            color,
            line_width,
        )

    banner = f"{source_label}  |  {summary.capsule_count} detected  |  {fps:.1f} inference FPS"
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (20, 24, 31), -1)
    cv2.putText(
        output,
        banner,
        (9, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


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
