from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


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
        return self.by_class.get("capsule", self.total)

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


def summarize_result(result: Any, target_class: str = "capsule") -> CountSummary:
    names = getattr(result, "names", {}) or {}
    detections = getattr(result, "obb", None)
    if detections is None:
        detections = getattr(result, "boxes", None)
    if detections is None or getattr(detections, "cls", None) is None:
        return CountSummary(total=0, by_class={target_class: 0})

    class_ids = _to_list(detections.cls)
    confidences = _to_list(getattr(detections, "conf", None))
    xywhr_rows = _to_list(getattr(detections, "xywhr", None))

    counts: dict[str, int] = {}
    measurements: list[CapsuleMeasurement] = []
    for index, class_id in enumerate(class_ids):
        class_name = names.get(int(class_id), str(int(class_id)))
        counts[class_name] = counts.get(class_name, 0) + 1

        if index < len(xywhr_rows):
            x_px, y_px, width_px, height_px, angle_rad = xywhr_rows[index]
            confidence = confidences[index] if index < len(confidences) else 0.0
            measurements.append(
                CapsuleMeasurement(
                    index=index + 1,
                    class_name=class_name,
                    confidence=float(confidence),
                    center_x_px=float(x_px),
                    center_y_px=float(y_px),
                    width_px=float(width_px),
                    height_px=float(height_px),
                    angle_deg=float(math.degrees(angle_rad)),
                )
            )

    counts.setdefault(target_class, 0)
    return CountSummary(total=sum(counts.values()), by_class=counts, measurements=measurements)
