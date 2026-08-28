import sys
import unittest
from unittest.mock import Mock, patch

import cv2

from src.capsule_yolo.video_source import open_video_capture, parse_video_source


class VideoSourceTests(unittest.TestCase):
    def test_csi_pipeline_uses_requested_resolution(self) -> None:
        spec = parse_video_source(
            "csi:1",
            width=1920,
            height=1080,
            framerate=30,
            exposure_us=8000,
            analog_gain=1.0,
            digital_gain=1.0,
        )

        self.assertEqual(spec.label, "csi:1")
        self.assertIn("sensor-id=1", spec.capture_source)
        self.assertIn("width=(int)1920", spec.capture_source)
        self.assertIn("height=(int)1080", spec.capture_source)
        self.assertIn("framerate=(fraction)30/1", spec.capture_source)
        self.assertIn('exposuretimerange="8000000 8000000"', spec.capture_source)
        self.assertIn('gainrange="1 1"', spec.capture_source)
        self.assertIn('ispdigitalgainrange="1 1"', spec.capture_source)

    @patch("src.capsule_yolo.video_source.cv2.VideoCapture")
    def test_numeric_camera_requests_high_resolution(self, video_capture: Mock) -> None:
        capture = video_capture.return_value

        returned_capture, spec = open_video_capture("0", width=1920, height=1080, framerate=30)

        self.assertIs(returned_capture, capture)
        self.assertEqual(spec.capture_source, 0)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        capture.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
        expected_auto_exposure = 0.25 if sys.platform == "win32" else 1.0
        expected_exposure = -7 if sys.platform == "win32" else 80.0
        capture.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, expected_auto_exposure)
        capture.set.assert_any_call(cv2.CAP_PROP_EXPOSURE, expected_exposure)
        capture.set.assert_any_call(cv2.CAP_PROP_GAIN, 1.0)

    @patch("src.capsule_yolo.video_source.cv2.VideoCapture")
    def test_zero_exposure_restores_camera_auto_exposure(self, video_capture: Mock) -> None:
        capture = video_capture.return_value

        open_video_capture("0", exposure_us=0)

        expected_auto_exposure = 0.75 if sys.platform == "win32" else 3.0
        capture.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, expected_auto_exposure)

    @patch("src.capsule_yolo.video_source.cv2.VideoCapture")
    def test_video_file_is_not_forced_to_camera_resolution(self, video_capture: Mock) -> None:
        capture = video_capture.return_value

        open_video_capture("data/samples/test.mp4", width=1920, height=1080, framerate=30)

        capture.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
