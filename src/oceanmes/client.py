from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import requests

from .models import InspectionManifest, ServerDeviceConfiguration
from .settings import OceanMesSettings


class OceanMesError(RuntimeError):
    """Base exception for the edge transport boundary."""


class OceanMesTransportError(OceanMesError):
    """No HTTP response was received; the exact request can be retried."""


@dataclass
class OceanMesResponseError(OceanMesError):
    status_code: int
    error_code: str
    detail: str
    retryable: bool

    def __str__(self) -> str:
        return f"OCEANMES HTTP {self.status_code} ({self.error_code}): {self.detail}"


class OceanMesClient:
    """Synchronous transport intended for a future background outbox worker.

    No POST is retried implicitly. The durable outbox will retry the exact same
    UUID, canonical manifest, and evidence explicitly.
    """

    def __init__(
        self,
        settings: OceanMesSettings,
        *,
        session: requests.Session | None = None,
    ) -> None:
        settings.validate()
        if not settings.enabled:
            raise ValueError("OCEANMES integration is disabled.")
        self.settings = settings
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {settings.api_key}",
                "Accept": "application/json",
                "User-Agent": f"capsule-jetson/{settings.edge_software_version}",
            }
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "OceanMesClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_configuration(self) -> ServerDeviceConfiguration:
        response = self._request("get", "/api/edge/v1/config")
        payload = self._success_payload(response, accepted={200})
        try:
            return ServerDeviceConfiguration.from_response(payload)
        except ValueError as exc:
            raise OceanMesError(str(exc)) from exc

    def upload_inspection(self, inspection: InspectionManifest) -> dict[str, Any]:
        with inspection.evidence_path.open("rb") as evidence:
            response = self._request(
                "post",
                "/api/edge/v1/inspections",
                files={
                    "manifest": (
                        "manifest.json",
                        io.BytesIO(inspection.canonical_bytes()),
                        "application/json",
                    ),
                    "evidence": (
                        inspection.evidence_path.name,
                        evidence,
                        "image/jpeg",
                    ),
                },
            )
        payload = self._success_payload(response, accepted={200, 201})
        if payload.get("edge_inspection_id") != inspection.edge_inspection_id:
            raise OceanMesError("OCEANMES acknowledged a different inspection identifier.")
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            return self._session.request(
                method,
                f"{self.settings.base_url}{path}",
                timeout=self.settings.timeout,
                verify=self.settings.requests_verify,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise OceanMesTransportError(
                f"OCEANMES request failed before a response: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _success_payload(
        response: requests.Response, *, accepted: set[int]
    ) -> dict[str, Any]:
        try:
            payload = response.json()
        except (TypeError, ValueError):
            payload = {}
        if response.status_code not in accepted or not payload.get("ok"):
            status = int(response.status_code)
            raise OceanMesResponseError(
                status_code=status,
                error_code=str(payload.get("error") or "invalid_response"),
                detail=str(
                    payload.get("detail") or "The server returned an invalid response."
                ),
                retryable=status in {408, 425, 429} or status >= 500,
            )
        return payload
