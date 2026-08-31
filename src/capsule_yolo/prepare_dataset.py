from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DATA_YAML,
    IMAGE_SUFFIXES,
    LABELED_DATA_DIR,
    PREPARED_DATA_DIR,
    PROJECT_ROOT,
    project_path,
)


@dataclass(frozen=True)
class DatasetSplit:
    train: float = 0.74
    val: float = 0.16
    test: float = 0.10

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must add to 1.0, got {total:.3f}")


def load_class_names(source_dir: Path) -> list[str]:
    classes_path = source_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class list: {classes_path}")

    class_names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"No class names found in {classes_path}")
    if len(class_names) != len(set(class_names)):
        raise ValueError(f"Duplicate class names found in {classes_path}")
    return class_names


def find_pairs(source_dir: Path, class_count: int) -> list[tuple[Path, Path]]:
    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)

    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}
    missing_images = sorted(label_stems - image_stems)

    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"images missing labels: {', '.join(missing_labels)}")
        if missing_images:
            details.append(f"labels missing images: {', '.join(missing_images)}")
        raise ValueError("; ".join(details))

    if not pairs:
        raise ValueError(f"No labeled image pairs found in {source_dir}")

    validate_obb_labels([label_path for _, label_path in pairs], class_count)
    return pairs


def parse_obb_row(
    label_path: Path, line_number: int, line: str, class_count: int
) -> tuple[int, list[float]]:
    columns = line.split()
    location = f"{label_path.name}:{line_number}"
    if len(columns) != 9:
        raise ValueError(f"{location} has {len(columns)} columns; expected 9")

    try:
        class_id = int(columns[0])
    except ValueError as exc:
        raise ValueError(f"{location} has a non-integer class ID: {columns[0]}") from exc
    if not 0 <= class_id < class_count:
        raise ValueError(f"{location} has class ID {class_id}; expected 0 through {class_count - 1}")

    try:
        coordinates = [float(value) for value in columns[1:]]
    except ValueError as exc:
        raise ValueError(f"{location} contains a non-numeric coordinate") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"{location} contains a non-finite coordinate")

    clipped = [min(1.0, max(0.0, value)) for value in coordinates]
    points = list(zip(clipped[0::2], clipped[1::2]))
    twice_area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    )
    if twice_area <= 1e-12:
        raise ValueError(f"{location} becomes a zero-area polygon after boundary clipping")
    return class_id, coordinates


def validate_obb_labels(label_paths: list[Path], class_count: int) -> None:
    bad_rows: list[str] = []
    for label_path in label_paths:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parse_obb_row(label_path, line_number, line, class_count)
            except ValueError as exc:
                bad_rows.append(str(exc))
    if bad_rows:
        raise ValueError(
            "OBB labels must use 'class x1 y1 x2 y2 x3 y3 x4 y4'. "
            f"Invalid rows: {'; '.join(bad_rows[:10])}"
        )


def split_pairs(
    pairs: list[tuple[Path, Path]],
    split: DatasetSplit,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    split.validate()
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, round(total * split.train))
    val_count = max(1, round(total * split.val)) if total >= 3 else 0

    if train_count + val_count >= total and total >= 3:
        train_count = total - 2
        val_count = 1

    test_count = total - train_count - val_count
    if total >= 3 and test_count == 0:
        test_count = 1
        train_count -= 1

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def reset_output_dirs(output_dir: Path) -> None:
    for cache_path in (output_dir / "labels").glob("*.cache"):
        cache_path.unlink()
    for subset in ("train", "val", "test"):
        for kind in ("images", "labels"):
            target = output_dir / kind / subset
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def write_clipped_label(source: Path, target: Path, class_count: int) -> tuple[int, int]:
    output_rows: list[str] = []
    clipped_coordinates = 0
    clipped_boxes = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        class_id, coordinates = parse_obb_row(source, line_number, line, class_count)
        clipped = [min(1.0, max(0.0, value)) for value in coordinates]
        changed = sum(original != bounded for original, bounded in zip(coordinates, clipped))
        clipped_coordinates += changed
        clipped_boxes += int(changed > 0)
        output_rows.append(" ".join([str(class_id), *(f"{value:.10g}" for value in clipped)]))

    target.write_text("\n".join(output_rows) + "\n", encoding="utf-8")
    return clipped_boxes, clipped_coordinates


def copy_split_files(
    splits: dict[str, list[tuple[Path, Path]]], output_dir: Path, class_count: int
) -> tuple[int, int]:
    reset_output_dirs(output_dir)
    clipped_boxes = 0
    clipped_coordinates = 0
    for subset, pairs in splits.items():
        for image_path, label_path in pairs:
            shutil.copy2(image_path, output_dir / "images" / subset / image_path.name)
            boxes, coordinates = write_clipped_label(
                label_path,
                output_dir / "labels" / subset / label_path.name,
                class_count,
            )
            clipped_boxes += boxes
            clipped_coordinates += coordinates
    return clipped_boxes, clipped_coordinates


def count_boxes(label_paths: list[Path]) -> int:
    total = 0
    for label_path in label_paths:
        total += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def write_data_yaml(output_dir: Path, data_yaml: Path, class_names: list[str]) -> None:
    resolved_output = output_dir.resolve()
    try:
        yaml_path = resolved_output.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        yaml_path = resolved_output.as_posix()
    names = [
        f"  {class_id}: {json.dumps(class_name)}"
        for class_id, class_name in enumerate(class_names)
    ]
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {yaml_path}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                *names,
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_dataset(source_dir: Path, output_dir: Path, seed: int, split: DatasetSplit) -> None:
    class_names = load_class_names(source_dir)
    pairs = find_pairs(source_dir, len(class_names))
    splits = split_pairs(pairs, split, seed)
    clipped_boxes, clipped_coordinates = copy_split_files(splits, output_dir, len(class_names))
    write_data_yaml(output_dir, DATA_YAML, class_names)

    print(f"Prepared dataset from {source_dir}")
    print(f"Classes: {', '.join(f'{index}={name}' for index, name in enumerate(class_names))}")
    for subset, subset_pairs in splits.items():
        labels = [label_path for _, label_path in subset_pairs]
        print(f"{subset}: {len(subset_pairs)} images, {count_boxes(labels)} boxes")
    if clipped_coordinates:
        print(
            f"Boundary clipping: {clipped_coordinates} coordinates in {clipped_boxes} boxes "
            "(prepared labels only)"
        )
    print(f"Output: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare uploaded YOLO labels for training.")
    parser.add_argument("--source", default=str(LABELED_DATA_DIR), help="Source labeled_data directory.")
    parser.add_argument("--output", default=str(PREPARED_DATA_DIR), help="Prepared dataset output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible splits.")
    parser.add_argument("--train", type=float, default=0.74, help="Training split ratio.")
    parser.add_argument("--val", type=float, default=0.16, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split = DatasetSplit(train=args.train, val=args.val, test=args.test)
    prepare_dataset(project_path(args.source), project_path(args.output), args.seed, split)


if __name__ == "__main__":
    main()
