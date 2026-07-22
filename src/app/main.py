from __future__ import annotations

import os
import sys
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

from src.capsule_yolo.config import DEFAULT_MODEL_IMGSZ, DEFAULT_TRAINED_MODEL, PROJECT_ROOT
from src.app.video_worker import CounterSettings, VideoWorker


app = FastAPI(title="YOLO Capsule Counter")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

worker = VideoWorker()
settings = CounterSettings(
    model=os.getenv("CAPSULE_MODEL", str(DEFAULT_TRAINED_MODEL)),
    source=os.getenv("CAPSULE_SOURCE", "0"),
    imgsz=int(os.getenv("CAPSULE_IMGSZ", str(DEFAULT_MODEL_IMGSZ))),
    conf=float(os.getenv("CAPSULE_CONF", "0.25")),
    iou=float(os.getenv("CAPSULE_IOU", "0.7")),
    device=os.getenv("CAPSULE_DEVICE", "0") or None,
    capture_width=int(os.getenv("CAPSULE_CAPTURE_WIDTH", "1920")),
    capture_height=int(os.getenv("CAPSULE_CAPTURE_HEIGHT", "1080")),
    capture_fps=int(os.getenv("CAPSULE_CAPTURE_FPS", "30")),
    jpeg_quality=int(os.getenv("CAPSULE_JPEG_QUALITY", "95")),
    exposure_us=int(os.getenv("CAPSULE_EXPOSURE_US", "8000")),
    analog_gain=float(os.getenv("CAPSULE_ANALOG_GAIN", "1.0")),
    digital_gain=float(os.getenv("CAPSULE_DIGITAL_GAIN", "1.0")),
    half=os.getenv("CAPSULE_HALF", "true").strip().lower() in {"1", "true", "yes", "on"},
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "settings": settings,
            "project_root": PROJECT_ROOT,
            "model_exists": Path(settings.model).exists(),
        },
    )


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    worker.start(settings)
    return StreamingResponse(worker.frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats() -> JSONResponse:
    payload = worker.stats()
    payload["model_exists"] = Path(settings.model).exists()
    return JSONResponse(payload)


@app.post("/settings")
async def update_settings(request: Request) -> JSONResponse:
    global settings
    payload = await request.json()
    settings = CounterSettings(
        model=str(payload.get("model") or settings.model),
        source=str(payload.get("source") or settings.source),
        imgsz=int(payload.get("imgsz") or settings.imgsz),
        conf=float(payload.get("conf") or settings.conf),
        iou=float(payload.get("iou") or settings.iou),
        device=(str(payload.get("device")) if payload.get("device") else None),
        capture_width=int(payload.get("capture_width") or settings.capture_width),
        capture_height=int(payload.get("capture_height") or settings.capture_height),
        capture_fps=int(payload.get("capture_fps") or settings.capture_fps),
        jpeg_quality=settings.jpeg_quality,
        exposure_us=(
            int(payload["exposure_us"]) if "exposure_us" in payload else settings.exposure_us
        ),
        analog_gain=(
            float(payload["analog_gain"]) if "analog_gain" in payload else settings.analog_gain
        ),
        digital_gain=(
            float(payload["digital_gain"]) if "digital_gain" in payload else settings.digital_gain
        ),
        half=bool(payload.get("half", settings.half)),
    )
    worker.restart(settings)
    return JSONResponse({"ok": True, "settings": settings.__dict__})


@app.post("/stop")
def stop() -> JSONResponse:
    worker.stop()
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app.main:app", host="127.0.0.1", port=8000, reload=False)
