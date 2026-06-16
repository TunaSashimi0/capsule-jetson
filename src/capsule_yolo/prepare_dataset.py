from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_YAML, IMAGE_SUFFIXES, LABELED_DATA_DIR, PREPARED_DATA_DIR, project_path


@dataclass(frozen=True)
class DatasetSplit:
    train: float = 0.74
    val: float = 0.16
    test: float = 0.10

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must add to 1.0, got {total:.3f}")


def find_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
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

    validate_obb_labels([label_path for _, label_path in pairs])
    return pairs


def validate_obb_labels(label_paths: list[Path]) -> None:
    bad_rows: list[str] = []
    for label_path in label_paths:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) != 9:
                bad_rows.append(f"{label_path.name}:{line_number} has {len(columns)} columns")
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


def copy_split_files(splits: dict[str, list[tuple[Path, Path]]], output_dir: Path) -> None:
    reset_output_dirs(output_dir)
    for subset, pairs in splits.items():
        for image_path, label_path in pairs:
            shutil.copy2(image_path, output_dir / "images" / subset / image_path.name)
            shutil.copy2(label_path, output_dir / "labels" / subset / label_path.name)


def count_boxes(label_paths: list[Path]) -> int:
    total = 0
    for label_path in label_paths:
        total += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def write_data_yaml(output_dir: Path, data_yaml: Path) -> None:
    yaml_path = output_dir.resolve().as_posix()
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
                "  0: capsule",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_dataset(source_dir: Path, output_dir: Path, seed: int, split: DatasetSplit) -> None:
    pairs = find_pairs(source_dir)
    splits = split_pairs(pairs, split, seed)
    copy_split_files(splits, output_dir)
    write_data_yaml(output_dir, DATA_YAML)

    print(f"Prepared dataset from {source_dir}")
    for subset, subset_pairs in splits.items():
        labels = [label_path for _, label_path in subset_pairs]
        print(f"{subset}: {len(subset_pairs)} images, {count_boxes(labels)} boxes")
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
