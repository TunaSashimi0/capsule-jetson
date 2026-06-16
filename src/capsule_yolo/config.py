from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_DATA_DIR = PROJECT_ROOT / "labeled_data"
PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "prepared"
DATA_YAML = PROJECT_ROOT / "configs" / "data" / "capsule.yaml"
DEFAULT_BASE_MODEL = "yolo11n-obb.pt"
DEFAULT_TRAINED_MODEL = PROJECT_ROOT / "runs" / "train" / "capsule_yolo11n_obb" / "weights" / "best.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source
