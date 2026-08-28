from types import SimpleNamespace
import unittest

import numpy as np

from src.capsule_yolo.counting import summarize_result


def make_result(rows: list[list[float]], classes: list[int], confidences: list[float]) -> SimpleNamespace:
    detections = SimpleNamespace(
        cls=np.asarray(classes, dtype=np.float32),
        conf=np.asarray(confidences, dtype=np.float32),
        xywhr=np.asarray(rows, dtype=np.float32),
    )
    return SimpleNamespace(
        names={0: "capsule_defect", 1: "capsule_good"},
        obb=detections,
    )


class SummarizeResultTests(unittest.TestCase):
    def test_defect_wins_over_higher_confidence_good_detection(self) -> None:
        result = make_result(
            rows=[[100, 100, 40, 20, 0], [101, 100, 42, 20, 0]],
            classes=[1, 0],
            confidences=[0.95, 0.40],
        )

        summary = summarize_result(result)

        self.assertEqual(summary.total, 1)
        self.assertEqual(summary.good_count, 0)
        self.assertEqual(summary.defect_count, 1)
        self.assertEqual(summary.measurements[0].class_name, "capsule_defect")
        self.assertAlmostEqual(summary.measurements[0].confidence, 0.40, places=5)

    def test_separate_capsules_keep_separate_labels(self) -> None:
        result = make_result(
            rows=[[100, 100, 40, 20, 0], [200, 100, 40, 20, 0]],
            classes=[1, 0],
            confidences=[0.90, 0.80],
        )

        summary = summarize_result(result)

        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.good_count, 1)
        self.assertEqual(summary.defect_count, 1)

    def test_duplicate_good_detections_use_highest_confidence(self) -> None:
        result = make_result(
            rows=[[100, 100, 40, 20, 0], [101, 100, 40, 20, 0]],
            classes=[1, 1],
            confidences=[0.60, 0.90],
        )

        summary = summarize_result(result)

        self.assertEqual(summary.total, 1)
        self.assertEqual(summary.good_count, 1)
        self.assertAlmostEqual(summary.measurements[0].confidence, 0.90, places=5)


if __name__ == "__main__":
    unittest.main()
