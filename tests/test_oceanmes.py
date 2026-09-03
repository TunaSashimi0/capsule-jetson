import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.oceanmes.client import OceanMesClient, OceanMesResponseError
from src.oceanmes.models import (
    CameraInspectionSummary,
    InspectionManifest,
    ModelIdentity,
    ServerDeviceConfiguration,
)
from src.oceanmes.settings import OceanMesConfigurationError, OceanMesSettings


TEST_KEY = "oce_edge_" + ("x" * 43)


def valid_settings(**changes):
    values = {
        "enabled": True,
        "base_url": "https://oceanmes.example",
        "api_key": TEST_KEY,
        "edge_software_version": "test-r1",
    }
    values.update(changes)
    return OceanMesSettings(**values)


def server_configuration():
    return ServerDeviceConfiguration(
        device_name="Capsule Jetson 1",
        configuration_version=4,
        configuration_updated_at="2026-09-03T12:00:00Z",
        production_line="Capsule Line 1",
        room_id=12,
        room_code="CAP-101",
        room_name="Capsule Room 101",
    )


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def request(self, method, url, **kwargs):
        snapshot = {"method": method, "url": url, **kwargs}
        if "files" in snapshot:
            manifest_file = snapshot["files"]["manifest"][1]
            evidence_file = snapshot["files"]["evidence"][1]
            snapshot["manifest_bytes"] = manifest_file.read()
            snapshot["evidence_bytes"] = evidence_file.read()
        self.requests.append(snapshot)
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class OceanMesSettingsTests(unittest.TestCase):
    def test_disabled_is_the_safe_default(self):
        settings = OceanMesSettings.from_env({})
        self.assertFalse(settings.enabled)

    def test_enabled_connection_requires_explicit_http_opt_in(self):
        with self.assertRaises(OceanMesConfigurationError):
            OceanMesSettings.from_env(
                {
                    "OCEANMES_ENABLED": "true",
                    "OCEANMES_BASE_URL": "http://192.168.1.10:5001",
                    "OCEANMES_EDGE_API_KEY": TEST_KEY,
                }
            )

    def test_api_key_is_not_in_settings_repr(self):
        self.assertNotIn(TEST_KEY, repr(valid_settings()))


class OceanMesManifestTests(unittest.TestCase):
    def test_builds_server_v1_manifest_and_hashes_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "inspection.jpg"
            evidence.write_bytes(b"labeled-jpeg-test")
            manifest = InspectionManifest.build(
                evidence_path=evidence,
                configuration=server_configuration(),
                result="defective",
                capsule_count=3,
                defect_count=1,
                max_defect_confidence_percentage=94.276,
                camera_count=2,
                camera_summaries=(
                    CameraInspectionSummary(0, 2, 1),
                    CameraInspectionSummary(1, 1, 0),
                ),
                model=ModelIdentity("capsule-yolo", "a" * 64),
                edge_software_version="test-r1",
                inspection_unix_seconds=100,
                capture_unix_seconds=110,
                edge_inspection_id="73b2c01b-f186-4bd4-b4d7-f8b3cb834d76",
            )

        payload = json.loads(manifest.canonical_bytes())
        self.assertEqual(payload["payload_version"], 1)
        self.assertEqual(payload["configuration_version"], 4)
        self.assertEqual(payload["max_defect_confidence_percentage"], 94.28)
        self.assertEqual(
            payload["evidence"]["sha256"],
            hashlib.sha256(b"labeled-jpeg-test").hexdigest(),
        )

    def test_rejects_inconsistent_camera_totals_before_network_io(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "inspection.jpg"
            evidence.write_bytes(b"jpeg")
            with self.assertRaisesRegex(ValueError, "Camera summaries"):
                InspectionManifest.build(
                    evidence_path=evidence,
                    configuration=server_configuration(),
                    result="normal",
                    capsule_count=2,
                    defect_count=0,
                    max_defect_confidence_percentage=None,
                    camera_count=1,
                    camera_summaries=(CameraInspectionSummary(0, 1, 0),),
                    model=ModelIdentity("capsule-yolo", "a" * 64),
                    edge_software_version="test-r1",
                )


class OceanMesClientTests(unittest.TestCase):
    def test_fetches_authoritative_server_configuration(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "ok": True,
                        "payload_version": 1,
                        "device": {
                            "device_name": "Capsule Jetson 1",
                            "configuration_version": 4,
                            "configuration_updated_at": "2026-09-03T12:00:00Z",
                            "production_line": "Capsule Line 1",
                            "room": {
                                "room_id": 12,
                                "room_code": "CAP-101",
                                "room_name": "Capsule Room 101",
                            },
                        },
                    },
                )
            ]
        )
        client = OceanMesClient(valid_settings(), session=session)

        configuration = client.get_configuration()

        self.assertEqual(configuration.room_code, "CAP-101")
        self.assertEqual(session.requests[0]["url"], "https://oceanmes.example/api/edge/v1/config")
        self.assertEqual(session.headers["Authorization"], f"Bearer {TEST_KEY}")

    def test_uploads_exact_manifest_and_evidence_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "inspection.jpg"
            evidence.write_bytes(b"jpeg-evidence")
            manifest = InspectionManifest.build(
                evidence_path=evidence,
                configuration=server_configuration(),
                result="normal",
                capsule_count=2,
                defect_count=0,
                max_defect_confidence_percentage=None,
                camera_count=1,
                camera_summaries=(CameraInspectionSummary(0, 2, 0),),
                model=ModelIdentity("capsule-yolo", "a" * 64),
                edge_software_version="test-r1",
            )
            session = FakeSession(
                [
                    FakeResponse(
                        201,
                        {
                            "ok": True,
                            "inspection_id": 42,
                            "edge_inspection_id": manifest.edge_inspection_id,
                            "duplicate": False,
                        },
                    )
                ]
            )
            client = OceanMesClient(valid_settings(), session=session)

            response = client.upload_inspection(manifest)

        request = session.requests[0]
        self.assertEqual(response["inspection_id"], 42)
        self.assertEqual(set(request["files"]), {"manifest", "evidence"})
        self.assertEqual(request["evidence_bytes"], b"jpeg-evidence")
        self.assertEqual(
            json.loads(request["manifest_bytes"])["edge_inspection_id"],
            manifest.edge_inspection_id,
        )

    def test_server_failure_is_classified_as_retryable(self):
        session = FakeSession(
            [FakeResponse(503, {"ok": False, "error": "service_unavailable", "detail": "retry"})]
        )
        client = OceanMesClient(valid_settings(), session=session)

        with self.assertRaises(OceanMesResponseError) as caught:
            client.get_configuration()

        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.error_code, "service_unavailable")


if __name__ == "__main__":
    unittest.main()
