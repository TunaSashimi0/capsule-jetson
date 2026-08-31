from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.app.settings import CounterSettingsUpdate
from src.app.video_worker import CounterSettings


class CounterSettingsTests(unittest.TestCase):
    def test_valid_partial_update_is_normalized(self) -> None:
        update = CounterSettingsUpdate(
            model="  model.pt  ",
            secondary_source="   ",
            imgsz=1280,
            exposure_us=0,
            autofocus=False,
        )

        self.assertEqual(update.model, "model.pt")
        self.assertIsNone(update.secondary_source)
        self.assertEqual(update.exposure_us, 0)
        self.assertFalse(update.autofocus)

    def test_image_size_must_match_model_stride(self) -> None:
        with self.assertRaises(ValidationError):
            CounterSettingsUpdate(imgsz=1000)

    def test_confidence_outside_probability_range_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CounterSettingsUpdate(conf=1.1)

    def test_unknown_settings_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CounterSettingsUpdate(unknown_option=True)  # type: ignore[call-arg]

    def test_runtime_settings_reject_blank_sources(self) -> None:
        settings = CounterSettings(model="model.pt", source="   ")

        with self.assertRaisesRegex(ValueError, "source must not be blank"):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
