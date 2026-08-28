from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_PATH = APP_DIR.parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.capsule_yolo.config import DEFAULT_TRAINED_MODEL, PROJECT_ROOT
from src.capsule_yolo.solenoid import SolenoidCycleController, SolenoidSettings
from src.app.video_worker import CounterSettings, VideoWorker


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


worker = VideoWorker()
settings = CounterSettings(
    model=os.getenv("CAPSULE_MODEL", str(DEFAULT_TRAINED_MODEL)),
    source=os.getenv("CAPSULE_SOURCE", "csi:0"),
    secondary_source=os.getenv("CAPSULE_SECONDARY_SOURCE", "csi:1") or None,
    imgsz=int(os.getenv("CAPSULE_IMGSZ", "1280")),
    conf=float(os.getenv("CAPSULE_CONF", "0.25")),
    iou=float(os.getenv("CAPSULE_IOU", "0.7")),
    device=os.getenv("CAPSULE_DEVICE", "0") or None,
    capture_width=int(os.getenv("CAPSULE_CAPTURE_WIDTH", "3280")),
    capture_height=int(os.getenv("CAPSULE_CAPTURE_HEIGHT", "2464")),
    capture_fps=int(os.getenv("CAPSULE_CAPTURE_FPS", "21")),
    preview_width=int(os.getenv("CAPSULE_PREVIEW_WIDTH", "1280")),
    preview_fps=float(os.getenv("CAPSULE_PREVIEW_FPS", "2")),
    preview_jpeg_quality=int(os.getenv("CAPSULE_PREVIEW_JPEG_QUALITY", "84")),
    autofocus=env_bool("CAPSULE_AUTOFOCUS", True),
)
solenoid_settings = SolenoidSettings(
    enabled=env_bool("CAPSULE_SOLENOID_ENABLED", False),
    chip=os.getenv("CAPSULE_SOLENOID_CHIP", "mcp23017"),
    bus=int(os.getenv("CAPSULE_SOLENOID_I2C_BUS", "7")),
    address=int(os.getenv("CAPSULE_SOLENOID_I2C_ADDRESS", "0x20"), 0),
    active_high=env_bool("CAPSULE_SOLENOID_ACTIVE_HIGH", True),
    intake_channel=int(os.getenv("CAPSULE_SOLENOID_INTAKE_CHANNEL", "0")),
    discharge_channel=int(os.getenv("CAPSULE_SOLENOID_DISCHARGE_CHANNEL", "1")),
    intake_seconds=float(os.getenv("CAPSULE_SOLENOID_INTAKE_SECONDS", "2")),
    inspection_seconds=float(os.getenv("CAPSULE_SOLENOID_INSPECTION_SECONDS", "30")),
    discharge_seconds=float(os.getenv("CAPSULE_SOLENOID_DISCHARGE_SECONDS", "3")),
    cooldown_seconds=float(os.getenv("CAPSULE_SOLENOID_COOLDOWN_SECONDS", "120")),
)
solenoid_controller = SolenoidCycleController(
    solenoid_settings,
    inference_snapshot=worker.stats,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.start(settings)
    solenoid_controller.start()
    yield
    solenoid_controller.stop()
    worker.stop()


app = FastAPI(title="YOLO Capsule Counter", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")
STYLE_PATH = APP_DIR / "static" / "styles.css"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "settings": settings,
            "project_root": PROJECT_ROOT,
            "model_exists": Path(settings.model).exists(),
            "asset_version": STYLE_PATH.stat().st_mtime_ns,
        },
    )
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    worker.start(settings)
    return StreamingResponse(
        worker.frames(0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/video_feed/{camera_index}")
def indexed_video_feed(camera_index: int) -> StreamingResponse:
    worker.start(settings)
    return StreamingResponse(
        worker.frames(camera_index),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/stats")
def stats() -> JSONResponse:
    payload = worker.stats()
    payload["model_exists"] = Path(settings.model).exists()
    payload["solenoid"] = solenoid_controller.stats()
    return JSONResponse(payload)


@app.post("/settings")
async def update_settings(request: Request) -> JSONResponse:
    global settings
    payload = await request.json()
    settings = CounterSettings(
        model=str(payload.get("model") or settings.model),
        source=str(payload.get("source") or settings.source),
        secondary_source=(str(payload.get("secondary_source") or "").strip() or None),
        imgsz=int(payload.get("imgsz") or settings.imgsz),
        conf=float(payload.get("conf") or settings.conf),
        iou=float(payload.get("iou") or settings.iou),
        device=(str(payload.get("device")) if payload.get("device") else None),
        capture_width=int(payload.get("capture_width") or settings.capture_width),
        capture_height=int(payload.get("capture_height") or settings.capture_height),
        capture_fps=int(payload.get("capture_fps") or settings.capture_fps),
        preview_width=int(payload.get("preview_width") or settings.preview_width),
        preview_fps=float(payload.get("preview_fps") or settings.preview_fps),
        preview_jpeg_quality=int(
            payload.get("preview_jpeg_quality") or settings.preview_jpeg_quality
        ),
        autofocus=str(payload.get("autofocus", "false")).lower() in {"1", "true", "yes", "on"},
    )
    solenoid_controller.stop()
    worker.restart(settings)
    solenoid_controller.start()
    return JSONResponse({"ok": True, "settings": settings.__dict__})


@app.post("/stop")
def stop() -> JSONResponse:
    solenoid_controller.stop()
    worker.stop()
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000, reload=False)
