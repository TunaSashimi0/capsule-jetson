from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Protocol


DEFAULT_MAX_FRAME_AGE_SECONDS = 5.0


@dataclass(frozen=True)
class SolenoidSettings:
    enabled: bool = False
    chip: str = "mcp23017"
    bus: int = 7
    address: int = 0x20
    active_high: bool = True
    intake_channel: int = 0
    discharge_channel: int = 1
    intake_seconds: float = 2.0
    inspection_seconds: float = 30.0
    discharge_seconds: float = 3.0
    cooldown_seconds: float = 120.0

    def validate(self) -> None:
        chip = self.chip.strip().lower()
        channel_limit = 16 if chip == "mcp23017" else 8 if chip == "mcp23008" else 0
        if not channel_limit:
            raise ValueError("chip must be mcp23017 or mcp23008")
        if self.intake_channel == self.discharge_channel:
            raise ValueError("intake and discharge channels must be different")
        for channel in (self.intake_channel, self.discharge_channel):
            if not 0 <= channel < channel_limit:
                raise ValueError(f"channel {channel} is invalid for {chip}")
        if self.intake_seconds <= 0 or self.inspection_seconds <= 0:
            raise ValueError("intake and inspection durations must be positive")
        if self.discharge_seconds <= 0 or self.cooldown_seconds < 0:
            raise ValueError("discharge must be positive and cooldown cannot be negative")
        if self.intake_seconds > self.inspection_seconds:
            raise ValueError("intake pulse cannot exceed the inspection window")


class SolenoidDriver(Protocol):
    def set_channel(self, channel: int, active: bool) -> None: ...

    def all_off(self) -> None: ...

    def close(self) -> None: ...


class MCP230xxSolenoidDriver:
    """Fail-safe two-channel output driver using Adafruit's MCP230xx library."""

    def __init__(self, settings: SolenoidSettings) -> None:
        settings.validate()
        from adafruit_extended_bus import ExtendedI2C

        chip = settings.chip.strip().lower()
        if chip == "mcp23017":
            from adafruit_mcp230xx.mcp23017 import MCP23017

            chip_class = MCP23017
        else:
            from adafruit_mcp230xx.mcp23008 import MCP23008

            chip_class = MCP23008

        self._settings = settings
        self._lock = threading.Lock()
        self._i2c = ExtendedI2C(settings.bus)
        self._mcp = chip_class(self._i2c, address=settings.address, reset=False)
        self._mask = (1 << settings.intake_channel) | (1 << settings.discharge_channel)

        # Set the output latches to the inactive level while the pins are still
        # inputs, then change only the two solenoid pins to outputs. This avoids
        # an activation pulse during initialization for either polarity.
        self._gpio_state = int(self._mcp.gpio)
        self._gpio_state = self._with_inactive_bits(self._gpio_state)
        self._mcp.gpio = self._gpio_state
        self._mcp.iodir = int(self._mcp.iodir) & ~self._mask
        self.all_off()

    def _with_inactive_bits(self, value: int) -> int:
        if self._settings.active_high:
            return value & ~self._mask
        return value | self._mask

    def set_channel(self, channel: int, active: bool) -> None:
        if channel not in (self._settings.intake_channel, self._settings.discharge_channel):
            raise ValueError(f"channel {channel} is not configured as a solenoid output")
        bit = 1 << channel
        electrical_high = active if self._settings.active_high else not active
        with self._lock:
            if electrical_high:
                self._gpio_state |= bit
            else:
                self._gpio_state &= ~bit
            self._mcp.gpio = self._gpio_state

    def all_off(self) -> None:
        with self._lock:
            self._gpio_state = self._with_inactive_bits(self._gpio_state)
            self._mcp.gpio = self._gpio_state

    def close(self) -> None:
        try:
            self.all_off()
        finally:
            deinit = getattr(self._i2c, "deinit", None)
            if callable(deinit):
                deinit()


@dataclass
class SolenoidCycleStats:
    enabled: bool = False
    status: str = "disabled"
    cycle_id: int = 0
    active_channel: int | None = None
    seconds_remaining: float = 0.0
    inspection_inferences: int = 0
    inspection_inferences_by_camera: list[int] | None = None
    chip: str = ""
    bus: int = 0
    address: str = ""
    active_high: bool = True
    error: str | None = None


class SolenoidCycleController:
    def __init__(
        self,
        settings: SolenoidSettings,
        *,
        inference_snapshot: Callable[[], dict[str, object]],
        driver_factory: Callable[[SolenoidSettings], SolenoidDriver] = MCP230xxSolenoidDriver,
        max_frame_age_seconds: float = DEFAULT_MAX_FRAME_AGE_SECONDS,
    ) -> None:
        settings.validate()
        if max_frame_age_seconds <= 0:
            raise ValueError("max_frame_age_seconds must be positive")
        self._settings = settings
        self._inference_snapshot = inference_snapshot
        self._driver_factory = driver_factory
        self._max_frame_age_seconds = max_frame_age_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._driver: SolenoidDriver | None = None
        self._phase_deadline = 0.0
        self._inspection_start_counts: dict[int, int] | None = None
        self._minimum_start_counts: dict[int, int] | None = None
        self._stats = SolenoidCycleStats(
            enabled=settings.enabled,
            status="disabled" if not settings.enabled else "stopped",
            chip=settings.chip,
            bus=settings.bus,
            address=f"0x{settings.address:02x}",
            active_high=settings.active_high,
        )

    def start(self) -> None:
        with self._lock:
            if not self._settings.enabled:
                self._stats.status = "disabled"
                return
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._stats.error = None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        driver = self._driver
        if driver is not None:
            try:
                driver.all_off()
            except Exception:
                pass
        with self._lock:
            self._stats.active_channel = None
            self._stats.seconds_remaining = 0.0
            self._stats.status = "stopped" if self._settings.enabled else "disabled"
            self._phase_deadline = 0.0
            self._inspection_start_counts = None
            if not (self._thread and self._thread.is_alive()):
                self._thread = None

    def stats(self) -> dict[str, object]:
        with self._lock:
            payload = asdict(self._stats)
            deadline = self._phase_deadline
            inspection_start = self._inspection_start_counts
        payload["seconds_remaining"] = max(0.0, deadline - time.monotonic()) if deadline else 0.0
        if inspection_start is not None:
            current = self._camera_inference_counts(self._inference_snapshot())
            counts = [
                max(0, current.get(index, 0) - start_count)
                for index, start_count in sorted(inspection_start.items())
            ]
            payload["inspection_inferences_by_camera"] = counts
            payload["inspection_inferences"] = sum(counts)
        return payload

    def _run(self) -> None:
        try:
            self._driver = self._driver_factory(self._settings)
            self._driver.all_off()
            while not self._stop_event.is_set():
                self._set_phase("waiting_for_inference")
                if not self._wait_for_inference():
                    break
                if not self._run_cycle():
                    continue
        except Exception as exc:
            self._safe_all_off()
            self._set_phase("error", error=str(exc))
        finally:
            driver = self._driver
            self._safe_all_off()
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass
            self._driver = None

    def _run_cycle(self) -> bool:
        assert self._driver is not None
        snapshot = self._inference_snapshot()
        start_counts = self._camera_inference_counts(snapshot)
        with self._lock:
            self._stats.cycle_id += 1
            self._stats.inspection_inferences = 0
            self._stats.inspection_inferences_by_camera = [0 for _ in start_counts]
            self._inspection_start_counts = start_counts

        inspection_deadline = time.monotonic() + self._settings.inspection_seconds
        self._driver.set_channel(self._settings.intake_channel, True)
        self._set_phase(
            "intake_open",
            channel=self._settings.intake_channel,
            deadline=time.monotonic() + self._settings.intake_seconds,
        )
        if not self._wait_phase(require_inference=True):
            self._finish_inspection()
            self._safe_all_off()
            return False

        self._driver.set_channel(self._settings.intake_channel, False)
        self._set_phase("inspecting", deadline=inspection_deadline)
        if not self._wait_phase(require_inference=True):
            self._finish_inspection()
            self._safe_all_off()
            return False

        if not self._finish_inspection():
            self._safe_all_off()
            self._set_phase("waiting_for_inference")
            return False
        self._driver.set_channel(self._settings.discharge_channel, True)
        self._set_phase(
            "discharge_open",
            channel=self._settings.discharge_channel,
            deadline=time.monotonic() + self._settings.discharge_seconds,
        )
        if not self._wait_phase(require_inference=False):
            self._safe_all_off()
            return False

        self._driver.set_channel(self._settings.discharge_channel, False)
        self._driver.all_off()
        self._set_phase(
            "cooldown",
            deadline=time.monotonic() + self._settings.cooldown_seconds,
        )
        return self._wait_phase(require_inference=False)

    def _wait_for_inference(self) -> bool:
        while not self._stop_event.is_set():
            snapshot = self._inference_snapshot()
            current = self._camera_inference_counts(snapshot)
            minimum = self._minimum_start_counts
            has_fresh_frames = minimum is None or (
                current.keys() == minimum.keys()
                and all(current[index] > minimum[index] for index in minimum)
            )
            if self._inference_ready(snapshot) and has_fresh_frames:
                return True
            if self._stop_event.wait(0.25):
                break
        return False

    def _wait_phase(self, *, require_inference: bool) -> bool:
        while not self._stop_event.is_set():
            with self._lock:
                remaining = self._phase_deadline - time.monotonic()
            if remaining <= 0:
                return True
            if require_inference and not self._inference_ready():
                self._set_phase("waiting_for_inference")
                return False
            if self._stop_event.wait(min(0.1, remaining)):
                return False
        return False

    def _inference_ready(self, snapshot: dict[str, object] | None = None) -> bool:
        snapshot = snapshot or self._inference_snapshot()
        cameras = snapshot.get("cameras", [])
        now = time.monotonic()
        return bool(
            snapshot.get("status") == "running"
            and cameras
            and all(
                camera.get("status") == "running"
                and isinstance(camera.get("frame_time"), (int, float))
                and 0.0 <= now - float(camera["frame_time"]) <= self._max_frame_age_seconds
                for camera in cameras
            )
        )

    @staticmethod
    def _camera_inference_counts(snapshot: dict[str, object]) -> dict[int, int]:
        cameras = snapshot.get("cameras", [])
        return {
            int(camera.get("camera_index", index)): int(camera.get("inference_count", 0))
            for index, camera in enumerate(cameras)
        }

    def _finish_inspection(self) -> bool:
        snapshot = self._inference_snapshot()
        inference_ready = self._inference_ready(snapshot)
        current = self._camera_inference_counts(snapshot)
        with self._lock:
            start = self._inspection_start_counts
            if start is None:
                return False
            counts = [
                max(0, current.get(index, 0) - start_count)
                for index, start_count in sorted(start.items())
            ]
            self._stats.inspection_inferences_by_camera = counts
            self._stats.inspection_inferences = sum(counts)
            self._inspection_start_counts = None
            self._minimum_start_counts = current
            return inference_ready and bool(counts) and all(count > 0 for count in counts)

    def _safe_all_off(self) -> None:
        driver = self._driver
        if driver is not None:
            try:
                driver.all_off()
            except Exception:
                pass

    def _set_phase(
        self,
        status: str,
        *,
        channel: int | None = None,
        deadline: float = 0.0,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._stats.status = status
            self._stats.active_channel = channel
            self._stats.error = error
            self._phase_deadline = deadline
