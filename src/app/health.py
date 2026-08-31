from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Mapping
from typing import Any


DEFAULT_STATS_URL = "http://127.0.0.1:8000/stats"
DEFAULT_MAX_FRAME_AGE_SECONDS = 10.0


def inference_is_ready(
    payload: Mapping[str, Any],
    *,
    max_frame_age_seconds: float = DEFAULT_MAX_FRAME_AGE_SECONDS,
    now: float | None = None,
) -> bool:
    """Return whether every configured camera has produced a recent inference."""
    if max_frame_age_seconds <= 0:
        raise ValueError("max_frame_age_seconds must be positive")
    if payload.get("status") != "running" or payload.get("model_exists") is not True:
        return False

    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        return False

    current_time = time.monotonic() if now is None else now
    for camera in cameras:
        if not isinstance(camera, Mapping) or camera.get("status") != "running":
            return False
        frame_time = camera.get("frame_time")
        if isinstance(frame_time, bool) or not isinstance(frame_time, (int, float)):
            return False
        frame_age = current_time - float(frame_time)
        if frame_age < 0 or frame_age > max_frame_age_seconds:
            return False
    return True


def check_local_stats(url: str = DEFAULT_STATS_URL) -> bool:
    with urllib.request.urlopen(url, timeout=2) as response:
        payload = json.load(response)
    return isinstance(payload, dict) and inference_is_ready(payload)


def main() -> int:
    try:
        return 0 if check_local_stats() else 1
    except (OSError, ValueError, json.JSONDecodeError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
