from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CountSummary:
    total: int
    by_class: dict[str, int] = field(default_factory=dict)

    @property
    def capsule_count(self) -> int:
        return self.by_class.get("capsule", self.total)


def summarize_result(result: Any, target_class: str = "capsule") -> CountSummary:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return CountSummary(total=0, by_class={target_class: 0})

    class_ids = boxes.cls
    if hasattr(class_ids, "detach"):
        class_ids = class_ids.detach()
    if hasattr(class_ids, "cpu"):
        class_ids = class_ids.cpu()
    if hasattr(class_ids, "tolist"):
        class_ids = class_ids.tolist()

    counts: dict[str, int] = {}
    for class_id in class_ids:
        class_name = names.get(int(class_id), str(int(class_id)))
        counts[class_name] = counts.get(class_name, 0) + 1

    counts.setdefault(target_class, 0)
    return CountSummary(total=sum(counts.values()), by_class=counts)
