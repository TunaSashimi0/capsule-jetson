from __future__ import annotations

import unittest

from src.app.health import inference_is_ready


class InferenceHealthTests(unittest.TestCase):
    @staticmethod
    def payload(*, frame_time: float = 99.0) -> dict[str, object]:
        return {
            "status": "running",
            "model_exists": True,
            "cameras": [
                {"camera_index": 0, "status": "running", "frame_time": frame_time},
                {"camera_index": 1, "status": "running", "frame_time": frame_time},
            ],
        }

    def test_recent_frames_from_all_cameras_are_ready(self) -> None:
        self.assertTrue(inference_is_ready(self.payload(), now=100.0))

    def test_stale_camera_makes_service_unhealthy(self) -> None:
        payload = self.payload()
        payload["cameras"][1]["frame_time"] = 80.0  # type: ignore[index]

        self.assertFalse(inference_is_ready(payload, now=100.0))

    def test_missing_model_is_unhealthy_even_when_http_endpoint_responds(self) -> None:
        payload = self.payload()
        payload["model_exists"] = False

        self.assertFalse(inference_is_ready(payload, now=100.0))

    def test_empty_camera_list_is_unhealthy(self) -> None:
        payload = self.payload()
        payload["cameras"] = []

        self.assertFalse(inference_is_ready(payload, now=100.0))

    def test_future_frame_timestamp_is_unhealthy(self) -> None:
        self.assertFalse(inference_is_ready(self.payload(frame_time=101.0), now=100.0))


if __name__ == "__main__":
    unittest.main()
