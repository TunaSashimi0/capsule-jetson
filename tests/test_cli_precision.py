from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.capsule_yolo.counting import CountSummary
from src.capsule_yolo import export_model, infer_video, predict_image, validate


class CliPrecisionTests(unittest.TestCase):
    def assert_half_argument(self, call: Mock) -> None:
        self.assertTrue(call.call_args.kwargs["half"])
        self.assertNotIn("quantize", call.call_args.kwargs)

    @patch("src.capsule_yolo.export_model.YOLO")
    def test_export_passes_supported_half_argument(self, yolo: Mock) -> None:
        with patch.object(sys, "argv", ["export-model", "--model", "model.pt"]):
            export_model.main()

        self.assert_half_argument(yolo.return_value.export)

    @patch("src.capsule_yolo.predict_image.YOLO")
    def test_image_prediction_passes_supported_half_argument(self, yolo: Mock) -> None:
        with patch.object(
            sys,
            "argv",
            ["predict-image", "--model", "model.pt", "--source", "image.jpg"],
        ):
            predict_image.main()

        self.assert_half_argument(yolo.return_value.predict)

    @patch("src.capsule_yolo.validate.YOLO")
    def test_validation_passes_supported_half_argument(self, yolo: Mock) -> None:
        with patch.object(sys, "argv", ["validate", "--model", "model.pt"]):
            validate.main()

        self.assert_half_argument(yolo.return_value.val)

    @patch("src.capsule_yolo.infer_video.annotated_frame")
    @patch("src.capsule_yolo.infer_video.summarize_result")
    @patch("src.capsule_yolo.infer_video.open_video_capture")
    @patch("src.capsule_yolo.infer_video.YOLO")
    def test_video_prediction_passes_supported_half_argument(
        self,
        yolo: Mock,
        open_capture: Mock,
        summarize: Mock,
        annotate: Mock,
    ) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, frame), (False, None)]
        open_capture.return_value = (capture, SimpleNamespace(label="test camera"))
        yolo.return_value.predict.return_value = [object()]
        summarize.return_value = CountSummary(total=0)
        annotate.return_value = frame

        with patch.object(
            sys,
            "argv",
            ["infer-video", "--model", "model.pt", "--source", "0", "--hide-window"],
        ), patch("builtins.print"):
            infer_video.main()

        self.assert_half_argument(yolo.return_value.predict)
        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
