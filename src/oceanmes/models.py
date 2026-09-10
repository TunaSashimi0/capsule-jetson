from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    sha256: str

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelIdentity":
        model_path = Path(path)
        if not model_path.is_file():
            raise ValueError(f"Model file does not exist: {model_path}")
        return cls(name=model_path.stem, sha256=sha256_file(model_path))


@dataclass(frozen=True)
class ServerDeviceConfiguration:
    device_name: str
    configuration_version: int
    configuration_updated_at: str
    production_line: str
    room_id: int
    room_code: str
    room_name: str

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> "ServerDeviceConfiguration":
        try:
            device = payload["device"]
            room = device["room"]
            result = cls(
                device_name=str(device["device_name"]),
                configuration_version=int(device["configuration_version"]),
                configuration_updated_at=str(device["configuration_updated_at"]),
                production_line=str(device["production_line"]),
                room_id=int(room["room_id"]),
                room_code=str(room["room_code"]),
                room_name=str(room["room_name"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("OCEANMES returned an invalid device configuration.") from exc
        if not payload.get("ok") or int(payload.get("payload_version", 0)) != 1:
            raise ValueError("OCEANMES returned an unsupported configuration payload.")
        if result.configuration_version < 1 or not result.room_code:
            raise ValueError("OCEANMES returned an incomplete device configuration.")
        return result


@dataclass(frozen=True)
class CameraInspectionSummary:
    camera_index: int
    capsule_count: int
    defect_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "camera_index": self.camera_index,
            "capsule_count": self.capsule_count,
            "defect_count": self.defect_count,
        }


@dataclass(frozen=True)
class InspectionManifest:
    payload: dict[str, Any]
    evidence_path: Path

    @classmethod
    def build(
        cls,
        *,
        evidence_path: str | Path,
        configuration: ServerDeviceConfiguration,
        result: str,
        capsule_count: int,
        defect_count: int,
        max_defect_confidence_percentage: float | None,
        camera_count: int,
        camera_summaries: Iterable[CameraInspectionSummary] | None,
        model: ModelIdentity,
        edge_software_version: str,
        inspection_unix_seconds: int | None = None,
        capture_unix_seconds: int | None = None,
        edge_inspection_id: str | None = None,
    ) -> "InspectionManifest":
        evidence = Path(evidence_path)
        if not evidence.is_file():
            raise ValueError(f"Evidence JPEG does not exist: {evidence}")
        if evidence.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("Pilot evidence must use a .jpg or .jpeg filename.")
        evidence_size = evidence.stat().st_size
        if evidence_size < 1 or evidence_size > MAX_EVIDENCE_BYTES:
            raise ValueError("Evidence JPEG must contain 1 byte through 16 MiB.")
        if result not in {"normal", "defective"}:
            raise ValueError("result must be normal or defective.")
        if capsule_count < 1 or defect_count < 0 or defect_count > capsule_count:
            raise ValueError("Capsule and defect counts are inconsistent.")
        if result == "normal" and (
            defect_count != 0 or max_defect_confidence_percentage is not None
        ):
            raise ValueError("A normal inspection must have no defects or defect confidence.")
        if result == "defective" and (
            defect_count < 1 or max_defect_confidence_percentage is None
        ):
            raise ValueError("A defective inspection requires a defect confidence.")
        if max_defect_confidence_percentage is not None and not (
            0 <= max_defect_confidence_percentage <= 100
        ):
            raise ValueError("Defect confidence must be from 0 through 100 percent.")
        if camera_count < 1:
            raise ValueError("camera_count must be at least one.")

        summaries = tuple(camera_summaries) if camera_summaries is not None else None
        if summaries is not None:
            if len(summaries) != camera_count:
                raise ValueError("Provide one camera summary per camera.")
            if len({item.camera_index for item in summaries}) != len(summaries):
                raise ValueError("Camera indexes must be unique.")
            if any(
                item.camera_index < 0
                or item.capsule_count < 0
                or item.defect_count < 0
                or item.defect_count > item.capsule_count
                for item in summaries
            ):
                raise ValueError("Camera summary counts are invalid.")
            if sum(item.capsule_count for item in summaries) != capsule_count or sum(
                item.defect_count for item in summaries
            ) != defect_count:
                raise ValueError("Camera summaries must equal the inspection totals.")

        capture_seconds = int(time.time()) if capture_unix_seconds is None else int(
            capture_unix_seconds
        )
        inspection_seconds = (
            capture_seconds
            if inspection_unix_seconds is None
            else int(inspection_unix_seconds)
        )
        if capture_seconds < 0 or inspection_seconds < 0:
            raise ValueError("Inspection timestamps must be nonnegative Unix seconds.")
        try:
            inspection_id = (
                str(uuid.UUID(edge_inspection_id))
                if edge_inspection_id
                else str(uuid.uuid4())
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("edge_inspection_id must be a UUID.") from exc

        payload: dict[str, Any] = {
            "payload_version": 1,
            "edge_inspection_id": inspection_id,
            "inspection_unix_seconds": inspection_seconds,
            "capture_unix_seconds": capture_seconds,
            "configuration_version": configuration.configuration_version,
            "result": result,
            "capsule_count": capsule_count,
            "defect_count": defect_count,
            "max_defect_confidence_percentage": (
                None
                if max_defect_confidence_percentage is None
                else round(float(max_defect_confidence_percentage), 2)
            ),
            "camera_count": camera_count,
            "model": {"name": model.name, "sha256": model.sha256},
            "edge_software_version": edge_software_version,
            "evidence": {
                "content_type": "image/jpeg",
                "byte_size": evidence_size,
                "sha256": sha256_file(evidence),
            },
        }
        if summaries is not None:
            payload["camera_summaries"] = [item.as_dict() for item in summaries]
        return cls(payload=payload, evidence_path=evidence)

    @property
    def edge_inspection_id(self) -> str:
        return str(self.payload["edge_inspection_id"])

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
