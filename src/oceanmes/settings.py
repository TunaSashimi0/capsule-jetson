from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


class OceanMesConfigurationError(ValueError):
    """The edge-to-OCEANMES runtime configuration is unsafe or incomplete."""


def _env_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise OceanMesConfigurationError(f"{name} must be true or false.")


def _env_float(
    values: Mapping[str, str], name: str, default: float, *, minimum: float
) -> float:
    try:
        result = float(values.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise OceanMesConfigurationError(f"{name} must be numeric.") from exc
    if result < minimum:
        raise OceanMesConfigurationError(f"{name} must be at least {minimum}.")
    return result


@dataclass(frozen=True)
class OceanMesSettings:
    """Connection settings loaded only from the device runtime environment."""

    enabled: bool = False
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 30.0
    heartbeat_seconds: float = 30.0
    verify_tls: bool = True
    ca_bundle: Path | None = None
    allow_http: bool = False
    edge_software_version: str = "capsule-jetson-dev"
    event_log_path: Path = Path("data/oceanmes/events.jsonl")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OceanMesSettings":
        values = os.environ if environ is None else environ
        ca_value = (values.get("OCEANMES_CA_BUNDLE") or "").strip()
        settings = cls(
            enabled=_env_bool(values, "OCEANMES_ENABLED", False),
            base_url=(values.get("OCEANMES_BASE_URL") or "").strip().rstrip("/"),
            api_key=(values.get("OCEANMES_EDGE_API_KEY") or "").strip(),
            connect_timeout_seconds=_env_float(
                values, "OCEANMES_CONNECT_TIMEOUT_SECONDS", 3.0, minimum=0.1
            ),
            read_timeout_seconds=_env_float(
                values, "OCEANMES_READ_TIMEOUT_SECONDS", 30.0, minimum=0.1
            ),
            heartbeat_seconds=_env_float(
                values, "OCEANMES_HEARTBEAT_SECONDS", 30.0, minimum=1.0
            ),
            verify_tls=_env_bool(values, "OCEANMES_VERIFY_TLS", True),
            ca_bundle=Path(ca_value).expanduser() if ca_value else None,
            allow_http=_env_bool(values, "OCEANMES_ALLOW_HTTP", False),
            edge_software_version=(
                values.get("CAPSULE_EDGE_SOFTWARE_VERSION") or "capsule-jetson-dev"
            ).strip(),
            event_log_path=Path(
                values.get("OCEANMES_EVENT_LOG") or "data/oceanmes/events.jsonl"
            ).expanduser(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.base_url:
            raise OceanMesConfigurationError(
                "OCEANMES_BASE_URL is required when OCEANMES_ENABLED=true."
            )
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OceanMesConfigurationError(
                "OCEANMES_BASE_URL must be an absolute http(s) URL."
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise OceanMesConfigurationError(
                "OCEANMES_BASE_URL cannot contain credentials, a query, or a fragment."
            )
        if parsed.scheme == "http" and not self.allow_http:
            raise OceanMesConfigurationError(
                "Plain HTTP is disabled. Use HTTPS or explicitly set "
                "OCEANMES_ALLOW_HTTP=true for an isolated LAN test."
            )
        if not self.api_key.startswith("oce_edge_") or len(self.api_key) <= 20:
            raise OceanMesConfigurationError(
                "OCEANMES_EDGE_API_KEY is missing or is not an edge-device key."
            )
        if len(self.edge_software_version) > 120 or not self.edge_software_version:
            raise OceanMesConfigurationError(
                "CAPSULE_EDGE_SOFTWARE_VERSION must contain 1-120 characters."
            )
        if self.ca_bundle is not None and not self.ca_bundle.is_file():
            raise OceanMesConfigurationError(
                f"OCEANMES_CA_BUNDLE does not exist: {self.ca_bundle}"
            )

    @property
    def requests_verify(self) -> bool | str:
        return str(self.ca_bundle) if self.ca_bundle is not None else self.verify_tls

    @property
    def timeout(self) -> tuple[float, float]:
        return self.connect_timeout_seconds, self.read_timeout_seconds
