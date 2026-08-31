from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CounterSettingsUpdate(BaseModel):
    """Validated subset of counter settings accepted by the web API."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, min_length=1, max_length=4096)
    source: str | None = Field(default=None, min_length=1, max_length=4096)
    secondary_source: str | None = Field(default=None, max_length=4096)
    imgsz: int | None = Field(default=None, ge=320, le=4096)
    conf: float | None = Field(default=None, gt=0.0, le=1.0)
    iou: float | None = Field(default=None, gt=0.0, le=1.0)
    device: str | None = Field(default=None, max_length=64)
    capture_width: int | None = Field(default=None, ge=320, le=8192)
    capture_height: int | None = Field(default=None, ge=240, le=8192)
    capture_fps: int | None = Field(default=None, ge=1, le=240)
    preview_width: int | None = Field(default=None, ge=320, le=3840)
    preview_fps: float | None = Field(default=None, gt=0.0, le=30.0)
    preview_jpeg_quality: int | None = Field(default=None, ge=30, le=95)
    autofocus: bool | None = None
    exposure_us: int | None = Field(default=None, ge=0, le=1_000_000)
    analog_gain: float | None = Field(default=None, ge=0.0, le=32.0)
    digital_gain: float | None = Field(default=None, ge=0.0, le=256.0)
    half: bool | None = None

    @field_validator("model", "source")
    @classmethod
    def require_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("secondary_source", "device")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("imgsz")
    @classmethod
    def require_model_stride(cls, value: int | None) -> int | None:
        if value is not None and value % 32:
            raise ValueError("must be divisible by 32")
        return value
