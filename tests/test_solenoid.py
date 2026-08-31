from __future__ import annotations

import threading
import time
import unittest

from src.capsule_yolo.solenoid import SolenoidCycleController, SolenoidSettings


class FakeDriver:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[float, str, int | None, bool | None]] = []

    def set_channel(self, channel: int, active: bool) -> None:
        with self.lock:
            self.events.append((time.monotonic(), "set", channel, active))

    def all_off(self) -> None:
        with self.lock:
            self.events.append((time.monotonic(), "all_off", None, None))

    def close(self) -> None:
        self.all_off()

    def set_events(self) -> list[tuple[float, int, bool]]:
        with self.lock:
            return [
                (at, int(channel), bool(active))
                for at, kind, channel, active in self.events
                if kind == "set"
            ]


class IncrementingInference:
    def __init__(self, *, stall_camera_1: bool = False) -> None:
        self.counts = [0, 0]
        self.stall_camera_1 = stall_camera_1
        self.lock = threading.Lock()

    def __call__(self) -> dict[str, object]:
        with self.lock:
            self.counts[0] += 1
            if not self.stall_camera_1:
                self.counts[1] += 1
            cameras = [
                {
                    "camera_index": index,
                    "status": "running",
                    "inference_count": count,
                    "frame_time": time.monotonic(),
                }
                for index, count in enumerate(self.counts)
            ]
            return {
                "status": "running",
                "cameras": cameras,
                "inference_count": sum(self.counts),
            }


def wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class SolenoidCycleControllerTests(unittest.TestCase):
    @staticmethod
    def settings() -> SolenoidSettings:
        return SolenoidSettings(
            enabled=True,
            intake_seconds=0.03,
            inspection_seconds=0.12,
            discharge_seconds=0.04,
            cooldown_seconds=0.08,
        )

    def test_cycle_uses_requested_timeline(self) -> None:
        driver = FakeDriver()
        controller = SolenoidCycleController(
            self.settings(),
            inference_snapshot=IncrementingInference(),
            driver_factory=lambda _: driver,
        )
        controller.start()
        try:
            self.assertTrue(
                wait_until(
                    lambda: (1, False) in [
                        (channel, active) for _, channel, active in driver.set_events()
                    ]
                )
            )
        finally:
            controller.stop()

        events = driver.set_events()[:4]
        self.assertEqual(
            [(channel, active) for _, channel, active in events],
            [(0, True), (0, False), (1, True), (1, False)],
        )
        self.assertAlmostEqual(events[1][0] - events[0][0], 0.03, delta=0.025)
        self.assertAlmostEqual(events[2][0] - events[0][0], 0.12, delta=0.03)
        self.assertAlmostEqual(events[3][0] - events[2][0], 0.04, delta=0.025)

    def test_discharge_stays_closed_when_one_camera_does_not_infer(self) -> None:
        driver = FakeDriver()
        controller = SolenoidCycleController(
            self.settings(),
            inference_snapshot=IncrementingInference(stall_camera_1=True),
            driver_factory=lambda _: driver,
        )
        controller.start()
        try:
            self.assertTrue(
                wait_until(
                    lambda: controller.stats()["status"] == "waiting_for_inference"
                    and controller.stats()["cycle_id"] == 1,
                    timeout=0.5,
                )
            )
            time.sleep(0.28)
        finally:
            controller.stop()

        events = driver.set_events()
        self.assertEqual(sum(channel == 0 and active for _, channel, active in events), 1)
        self.assertFalse(any(channel == 1 and active for _, channel, active in events))

    def test_disabled_controller_never_opens_the_i2c_driver(self) -> None:
        opened = False

        def factory(_: SolenoidSettings) -> FakeDriver:
            nonlocal opened
            opened = True
            return FakeDriver()

        controller = SolenoidCycleController(
            SolenoidSettings(enabled=False),
            inference_snapshot=IncrementingInference(),
            driver_factory=factory,
        )
        controller.start()
        controller.stop()
        self.assertFalse(opened)
        self.assertEqual(controller.stats()["status"], "disabled")

    def test_stale_camera_frame_is_not_inference_ready(self) -> None:
        stale_snapshot = {
            "status": "running",
            "cameras": [
                {
                    "camera_index": 0,
                    "status": "running",
                    "inference_count": 1,
                    "frame_time": time.monotonic() - 1.0,
                }
            ],
        }
        controller = SolenoidCycleController(
            SolenoidSettings(enabled=False),
            inference_snapshot=lambda: stale_snapshot,
            max_frame_age_seconds=0.05,
        )

        self.assertFalse(controller._inference_ready(stale_snapshot))

    def test_inspection_rejects_inference_count_from_a_stale_frame(self) -> None:
        stale_snapshot = {
            "status": "running",
            "cameras": [
                {
                    "camera_index": 0,
                    "status": "running",
                    "inference_count": 1,
                    "frame_time": time.monotonic() - 1.0,
                }
            ],
        }
        controller = SolenoidCycleController(
            SolenoidSettings(enabled=False),
            inference_snapshot=lambda: stale_snapshot,
            max_frame_age_seconds=0.05,
        )
        controller._inspection_start_counts = {0: 0}

        self.assertFalse(controller._finish_inspection())


if __name__ == "__main__":
    unittest.main()
