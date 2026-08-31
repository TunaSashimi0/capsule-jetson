from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.capsule_yolo.config import PREPARED_DATA_DIR
from src.capsule_yolo.prepare_dataset import (
    DatasetSplit,
    parse_obb_row,
    split_pairs,
    write_data_yaml,
)


class PrepareDatasetTests(unittest.TestCase):
    def test_valid_obb_row_is_parsed(self) -> None:
        class_id, coordinates = parse_obb_row(
            Path("sample.txt"),
            1,
            "1 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9",
            class_count=2,
        )

        self.assertEqual(class_id, 1)
        self.assertEqual(len(coordinates), 8)

    def test_zero_area_obb_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-area polygon"):
            parse_obb_row(
                Path("sample.txt"),
                1,
                "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4",
                class_count=2,
            )

    def test_split_is_deterministic_and_preserves_all_pairs(self) -> None:
        pairs = [(Path(f"{index}.jpg"), Path(f"{index}.txt")) for index in range(10)]

        first = split_pairs(pairs, DatasetSplit(), seed=42)
        second = split_pairs(pairs, DatasetSplit(), seed=42)

        self.assertEqual(first, second)
        self.assertEqual(sum(len(items) for items in first.values()), len(pairs))
        self.assertTrue(all(first[name] for name in ("train", "val", "test")))

    def test_project_dataset_yaml_uses_portable_relative_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=PREPARED_DATA_DIR.parent) as temp_dir:
            yaml_path = Path(temp_dir) / "capsule.yaml"
            write_data_yaml(PREPARED_DATA_DIR, yaml_path, ["defect", "good"])

            contents = yaml_path.read_text(encoding="utf-8")

        self.assertIn("path: data/prepared", contents)
        self.assertNotIn(":/", contents)


if __name__ == "__main__":
    unittest.main()
