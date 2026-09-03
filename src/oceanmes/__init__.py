"""OCEANMES edge-device transport boundary.

This package deliberately does not import the camera, PyTorch, or Ultralytics
stack. Network work can therefore run outside the inference hot path.
"""

from .client import OceanMesClient, OceanMesError, OceanMesResponseError
from .event_log import EdgeEventLog
from .heartbeat import OceanMesHeartbeat
from .models import (
    CameraInspectionSummary,
    InspectionManifest,
    ModelIdentity,
    ServerDeviceConfiguration,
)
from .settings import OceanMesConfigurationError, OceanMesSettings

__all__ = [
    "CameraInspectionSummary",
    "EdgeEventLog",
    "InspectionManifest",
    "ModelIdentity",
    "OceanMesClient",
    "OceanMesConfigurationError",
    "OceanMesError",
    "OceanMesHeartbeat",
    "OceanMesResponseError",
    "OceanMesSettings",
    "ServerDeviceConfiguration",
]
