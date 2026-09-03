from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_PATH = APP_DIR.parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.capsule_yolo.config import DEFAULT_MODEL_IMGSZ, DEFAULT_TRAINED_MODEL, PROJECT_ROOT
from src.capsule_yolo.solenoid import SolenoidCycleController, SolenoidSettings
from src.app.settings import CounterSettingsUpdate
from src.app.video_worker import CounterSettings, VideoWorker
from src.oceanmes import EdgeEventLog, OceanMesHeartbeat, OceanMesSettings


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
    imgsz=int(os.getenv("CAPSULE_IMGSZ", str(DEFAULT_MODEL_IMGSZ))),
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
    exposure_us=int(os.getenv("CAPSULE_EXPOSURE_US", "8000")),
    analog_gain=float(os.getenv("CAPSULE_ANALOG_GAIN", "1.0")),
    digital_gain=float(os.getenv("CAPSULE_DIGITAL_GAIN", "1.0")),
    half=env_bool("CAPSULE_HALF", True),
)
settings.validate()
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
oceanmes_settings = OceanMesSettings.from_env()
edge_event_log = EdgeEventLog(oceanmes_settings.event_log_path)
start_idle = env_bool("CAPSULE_START_IDLE", True)
start_inference_when_configured = env_bool(
    "OCEANMES_START_INFERENCE_WHEN_CONFIGURED", True
)
runtime_lock = threading.RLock()
runtime_state = "idle"
shutting_down = False


def _start_inference(reason: str) -> bool:
    global runtime_state
    with runtime_lock:
        if shutting_down or runtime_state in {"starting", "inference"}:
            return False
        runtime_state = "starting"
        try:
            worker.start(settings)
            solenoid_controller.start()
        except Exception as exc:
            runtime_state = "error"
            edge_event_log.emit(
                "inference_start_failed",
                reason=reason,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            raise
        runtime_state = "inference"
        edge_event_log.emit("inference_started", reason=reason)
        return True


def _stop_inference(reason: str) -> bool:
    global runtime_state
    with runtime_lock:
        if runtime_state in {"idle", "stopping"}:
            return False
        runtime_state = "stopping"
        solenoid_controller.stop()
        worker.stop()
        runtime_state = "idle"
        edge_event_log.emit("inference_stopped", reason=reason)
        return True


def _on_oceanmes_configured(configuration: Any) -> None:
    if start_inference_when_configured:
        _start_inference("oceanmes_configured")
        return
    edge_event_log.emit(
        "inference_held_idle",
        reason="automatic_start_disabled",
        configuration_version=configuration.configuration_version,
    )


oceanmes_heartbeat = OceanMesHeartbeat(
    oceanmes_settings,
    edge_event_log,
    on_configured=_on_oceanmes_configured,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global shutting_down
    with runtime_lock:
        shutting_down = False
    edge_event_log.emit(
        "edge_process_started",
        initial_state="idle" if start_idle else "local_inference",
    )
    if not start_idle:
        _start_inference("local_startup_configuration")
    oceanmes_heartbeat.start()
    try:
        yield
    finally:
        with runtime_lock:
            shutting_down = True
        oceanmes_heartbeat.stop()
        _stop_inference("process_shutdown")
        edge_event_log.emit("edge_process_stopped")
        edge_event_log.close()


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
    if runtime_state != "inference":
        raise HTTPException(status_code=409, detail="Inference is idle.")
    return StreamingResponse(
        worker.frames(0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/video_feed/{camera_index}")
def indexed_video_feed(camera_index: int) -> StreamingResponse:
    if runtime_state != "inference":
        raise HTTPException(status_code=409, detail="Inference is idle.")
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
    payload["runtime_state"] = runtime_state
    payload["unix_seconds"] = int(time.time())
    payload["oceanmes"] = oceanmes_heartbeat.status()
    return JSONResponse(payload)


@app.post("/settings")
async def update_settings(payload: CounterSettingsUpdate) -> JSONResponse:
    global runtime_state, settings
    updates = payload.model_dump(exclude_unset=True)

    def updated(name: str, current: Any) -> Any:
        value = updates.get(name)
        return current if value is None else value

    new_settings = CounterSettings(
        model=updated("model", settings.model),
        source=updated("source", settings.source),
        secondary_source=(
            updates["secondary_source"]
            if "secondary_source" in updates
            else settings.secondary_source
        ),
        imgsz=updated("imgsz", settings.imgsz),
        conf=updated("conf", settings.conf),
        iou=updated("iou", settings.iou),
        device=updates["device"] if "device" in updates else settings.device,
        capture_width=updated("capture_width", settings.capture_width),
        capture_height=updated("capture_height", settings.capture_height),
        capture_fps=updated("capture_fps", settings.capture_fps),
        preview_width=updated("preview_width", settings.preview_width),
        preview_fps=updated("preview_fps", settings.preview_fps),
        preview_jpeg_quality=updated(
            "preview_jpeg_quality", settings.preview_jpeg_quality
        ),
        autofocus=updated("autofocus", settings.autofocus),
        exposure_us=updated("exposure_us", settings.exposure_us),
        analog_gain=updated("analog_gain", settings.analog_gain),
        digital_gain=updated("digital_gain", settings.digital_gain),
        half=updated("half", settings.half),
    )
    new_settings.validate()
    with runtime_lock:
        runtime_state = "starting"
        try:
            solenoid_controller.stop()
            settings = new_settings
            worker.restart(settings)
            solenoid_controller.start()
        except Exception as exc:
            runtime_state = "error"
            edge_event_log.emit(
                "inference_restart_failed",
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            raise
        runtime_state = "inference"
    edge_event_log.emit("inference_started", reason="local_settings_update")
    return JSONResponse(
        {
            "ok": True,
            "runtime_state": runtime_state,
            "unix_seconds": int(time.time()),
            "settings": settings.__dict__,
        }
    )


@app.post("/stop")
def stop() -> JSONResponse:
    _stop_inference("local_stop_request")
    return JSONResponse(
        {
            "ok": True,
            "runtime_state": runtime_state,
            "unix_seconds": int(time.time()),
            "oceanmes_heartbeat_continues": oceanmes_settings.enabled,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000, reload=False)
