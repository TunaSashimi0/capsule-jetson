from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2


DEFECT_CLASS = "capsule_defect"
GOOD_CLASS = "capsule_good"
DEFAULT_CAPSULE_OVERLAP_THRESHOLD = 0.5


@dataclass(frozen=True)
class CapsuleMeasurement:
    index: int
    class_name: str
    confidence: float
    center_x_px: float
    center_y_px: float
    width_px: float
    height_px: float
    angle_deg: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class CountSummary:
    total: int
    by_class: dict[str, int] = field(default_factory=dict)
    measurements: list[CapsuleMeasurement] = field(default_factory=list)

    @property
    def capsule_count(self) -> int:
        return self.total

    @property
    def defect_count(self) -> int:
        return self.by_class.get(DEFECT_CLASS, 0)

    @property
    def good_count(self) -> int:
        return self.by_class.get(GOOD_CLASS, 0)

    @property
    def avg_width_px(self) -> float:
        return _average([item.width_px for item in self.measurements])

    @property
    def avg_height_px(self) -> float:
        return _average([item.height_px for item in self.measurements])

    @property
    def avg_angle_deg(self) -> float:
        return _average([item.angle_deg for item in self.measurements])

    def measurements_as_dicts(self, limit: int | None = None) -> list[dict[str, float | int | str]]:
        items = self.measurements if limit is None else self.measurements[:limit]
        return [item.to_dict() for item in items]


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


@dataclass(frozen=True)
class _DetectionCandidate:
    original_index: int
    class_name: str
    confidence: float
    center_x_px: float
    center_y_px: float
    width_px: float
    height_px: float
    angle_deg: float

    @property
    def rotated_rect(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return (
            (self.center_x_px, self.center_y_px),
            (self.width_px, self.height_px),
            self.angle_deg,
        )


def _class_name(names: dict[Any, Any], class_id: Any) -> str:
    numeric_class_id = int(class_id)
    return str(names.get(numeric_class_id, names.get(str(numeric_class_id), numeric_class_id)))


def _smaller_box_overlap(first: _DetectionCandidate, second: _DetectionCandidate) -> float:
    first_area = first.width_px * first.height_px
    second_area = second.width_px * second.height_px
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0:
        return 0.0

    _, intersection = cv2.rotatedRectangleIntersection(first.rotated_rect, second.rotated_rect)
    if intersection is None:
        return 0.0
    intersection_area = abs(float(cv2.contourArea(intersection)))
    return min(1.0, intersection_area / smaller_area)


def _cluster_candidates(
    candidates: list[_DetectionCandidate], overlap_threshold: float
) -> list[list[_DetectionCandidate]]:
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            if _smaller_box_overlap(candidates[first], candidates[second]) >= overlap_threshold:
                union(first, second)

    groups: dict[int, list[_DetectionCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)
    return sorted(groups.values(), key=lambda group: min(item.original_index for item in group))


def summarize_result(
    result: Any,
    target_class: str = "capsule",
    overlap_threshold: float = DEFAULT_CAPSULE_OVERLAP_THRESHOLD,
) -> CountSummary:
    names = getattr(result, "names", {}) or {}
    detections = getattr(result, "obb", None)
    if detections is None:
        detections = getattr(result, "boxes", None)
    if detections is None or getattr(detections, "cls", None) is None:
        return CountSummary(total=0, by_class={target_class: 0})

    class_ids = _to_list(detections.cls)
    confidences = _to_list(getattr(detections, "conf", None))
    xywhr_rows = _to_list(getattr(detections, "xywhr", None))
    if not xywhr_rows:
        xywh_rows = _to_list(getattr(detections, "xywh", None))
        xywhr_rows = [list(row[:4]) + [0.0] for row in xywh_rows]

    candidates: list[_DetectionCandidate] = []
    for index, class_id in enumerate(class_ids):
        if index < len(xywhr_rows):
            x_px, y_px, width_px, height_px, angle_rad = xywhr_rows[index]
            confidence = confidences[index] if index < len(confidences) else 0.0
            candidates.append(
                _DetectionCandidate(
                    original_index=index,
                    class_name=_class_name(names, class_id),
                    confidence=float(confidence),
                    center_x_px=float(x_px),
                    center_y_px=float(y_px),
                    width_px=float(width_px),
                    height_px=float(height_px),
                    angle_deg=float(math.degrees(angle_rad)),
                )
            )

    if not candidates:
        counts: dict[str, int] = {}
        for class_id in class_ids:
            class_name = _class_name(names, class_id)
            counts[class_name] = counts.get(class_name, 0) + 1
        counts.setdefault(target_class, 0)
        return CountSummary(total=sum(counts.values()), by_class=counts)

    counts: dict[str, int] = {}
    measurements: list[CapsuleMeasurement] = []
    for group in _cluster_candidates(candidates, overlap_threshold):
        defect_candidates = [item for item in group if item.class_name == DEFECT_CLASS]
        winner = max(defect_candidates or group, key=lambda item: item.confidence)
        class_name = DEFECT_CLASS if defect_candidates else winner.class_name
        counts[class_name] = counts.get(class_name, 0) + 1
        measurements.append(
            CapsuleMeasurement(
                index=len(measurements) + 1,
                class_name=class_name,
                confidence=winner.confidence,
                center_x_px=winner.center_x_px,
                center_y_px=winner.center_y_px,
                width_px=winner.width_px,
                height_px=winner.height_px,
                angle_deg=winner.angle_deg,
            )
        )

    counts.setdefault(target_class, 0)
    return CountSummary(total=len(measurements), by_class=counts, measurements=measurements)
