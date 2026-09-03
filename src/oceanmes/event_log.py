from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable


class EdgeEventLog:
    """Thread-safe JSONL event log with Unix timestamps and bounded rotation."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"capsule.oceanmes.events.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = RotatingFileHandler(
            self.path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        if not event or not event.strip():
            raise ValueError("event name is required")
        timestamp = self._clock()
        record: dict[str, Any] = {
            "unix_seconds": int(timestamp),
            "unix_milliseconds": int(timestamp * 1000),
            "utc": datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "event": event.strip(),
        }
        for name, value in fields.items():
            lowered = name.lower()
            if "api_key" in lowered or lowered in {"authorization", "password"}:
                raise ValueError(f"Refusing to log sensitive field: {name}")
            record[name] = value
        with self._lock:
            self._logger.info(
                json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
            )
        return record

    def close(self) -> None:
        with self._lock:
            handlers = list(self._logger.handlers)
            for handler in handlers:
                handler.flush()
                handler.close()
                self._logger.removeHandler(handler)
