from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Callable

from .client import OceanMesClient, OceanMesError, OceanMesResponseError
from .event_log import EdgeEventLog
from .models import ServerDeviceConfiguration
from .settings import OceanMesSettings


class OceanMesHeartbeat:
    """Background configuration heartbeat that never imports the ML stack."""

    def __init__(
        self,
        settings: OceanMesSettings,
        event_log: EdgeEventLog,
        *,
        on_configured: Callable[[ServerDeviceConfiguration], None] | None = None,
        client: OceanMesClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.event_log = event_log
        self.on_configured = on_configured
        self._clock = clock
        self._client = client
        self._owns_client = client is None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._configuration_fingerprint: tuple[object, ...] | None = None
        self._status: dict[str, object] = {
            "state": "disabled" if not settings.enabled else "idle",
            "heartbeat_seconds": settings.heartbeat_seconds,
            "last_attempt_unix_seconds": None,
            "last_success_unix_seconds": None,
            "consecutive_failures": 0,
            "last_error": None,
            "configuration": None,
        }

    def start(self) -> None:
        if not self.settings.enabled:
            self.event_log.emit("oceanmes_heartbeat_disabled")
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if self._client is None:
                self._client = OceanMesClient(self.settings)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="oceanmes-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(
                timeout=max(
                    2.0,
                    self.settings.connect_timeout_seconds
                    + self.settings.read_timeout_seconds
                    + 1.0,
                )
            )
        with self._lock:
            self._thread = None
            if self.settings.enabled:
                self._status["state"] = "stopped"
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
        self.event_log.emit("oceanmes_heartbeat_stopped")

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def poll_once(self) -> ServerDeviceConfiguration | None:
        if not self.settings.enabled:
            return None
        if self._client is None:
            self._client = OceanMesClient(self.settings)
        attempted_at = int(self._clock())
        with self._lock:
            self._status["state"] = "connecting"
            self._status["last_attempt_unix_seconds"] = attempted_at
        self.event_log.emit("oceanmes_heartbeat_attempt")
        try:
            configuration = self._client.get_configuration()
        except OceanMesError as exc:
            retryable = not isinstance(exc, OceanMesResponseError) or exc.retryable
            with self._lock:
                self._status["state"] = "error"
                self._status["consecutive_failures"] = (
                    int(self._status["consecutive_failures"]) + 1
                )
                self._status["last_error"] = {
                    "type": type(exc).__name__,
                    "retryable": retryable,
                    "message": str(exc),
                    "unix_seconds": attempted_at,
                }
            self.event_log.emit(
                "oceanmes_heartbeat_failed",
                error_type=type(exc).__name__,
                retryable=retryable,
                detail=str(exc),
            )
            return None

        succeeded_at = int(self._clock())
        configuration_payload = asdict(configuration)
        with self._lock:
            self._status["state"] = "connected"
            self._status["last_success_unix_seconds"] = succeeded_at
            self._status["consecutive_failures"] = 0
            self._status["last_error"] = None
            self._status["configuration"] = configuration_payload
        self.event_log.emit(
            "oceanmes_heartbeat_succeeded",
            device_name=configuration.device_name,
            configuration_version=configuration.configuration_version,
            production_line=configuration.production_line,
            room_code=configuration.room_code,
        )

        fingerprint = (
            configuration.device_name,
            configuration.configuration_version,
            configuration.production_line,
            configuration.room_id,
            configuration.room_code,
        )
        if fingerprint != self._configuration_fingerprint:
            self._configuration_fingerprint = fingerprint
            self.event_log.emit(
                "oceanmes_configuration_changed",
                device_name=configuration.device_name,
                configuration_version=configuration.configuration_version,
                production_line=configuration.production_line,
                room_code=configuration.room_code,
            )
            if self.on_configured is not None:
                try:
                    self.on_configured(configuration)
                except Exception as exc:
                    self.event_log.emit(
                        "oceanmes_configuration_callback_failed",
                        error_type=type(exc).__name__,
                        detail=str(exc),
                    )
        return configuration

    def _run(self) -> None:
        self.event_log.emit(
            "oceanmes_heartbeat_started",
            interval_seconds=self.settings.heartbeat_seconds,
        )
        while not self._stop_event.is_set():
            self.poll_once()
            if self._stop_event.wait(self.settings.heartbeat_seconds):
                break
