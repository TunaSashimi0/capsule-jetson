from __future__ import annotations

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

from src.capsule_yolo.config import DEFAULT_TRAINED_MODEL, PROJECT_ROOT
from src.app.video_worker import CounterSettings, VideoWorker


app = FastAPI(title="YOLO Capsule Counter")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

worker = VideoWorker()
settings = CounterSettings(model=str(DEFAULT_TRAINED_MODEL), source="0")


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
