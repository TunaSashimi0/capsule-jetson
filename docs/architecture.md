# Codebase architecture

## Purpose and scope

Capsule Jetson is an edge vision application for training and running a YOLO11 oriented-bounding-box (OBB) model. At runtime it captures one or two cameras, classifies capsules as good or defective, renders browser previews, and can optionally sequence two solenoid outputs through an MCP23008/MCP23017 I/O expander.

The repository deliberately keeps model weights, generated dataset splits, and run output outside Git. The checked-in source dataset is the input needed to reproduce those artifacts.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/capsule_yolo/` | Dataset preparation, training, validation, export, inference, counting, drawing, camera selection, autofocus, and solenoid control |
| `src/app/` | FastAPI application, request validation, video worker, health probe, templates, and static assets |
| `src/oceanmes/` | Server connection settings, authenticated HTTP transport, authoritative device configuration, and inspection manifest construction |
| `configs/` | Ultralytics dataset, training, and inference defaults |
| `tests/` | Hardware-independent unit and regression tests |
| `scripts/` | Jetson image/container helpers, camera overlay provisioning, and hardware diagnostics |
| `labeled_data/` | Canonical source images and OBB labels; images are stored with Git LFS |
| `data/prepared/` | Generated train/validation/test split; ignored by Git |
| `models/` and `runs/` | Local model artifacts and Ultralytics run output; ignored by Git |

## Runtime data flow

```text
CSI / USB / file source
        |
        v
VideoWorker capture loop (one native frame per configured camera)
        |
        +--> YOLO OBB inference --> overlap-aware capsule summary --> /stats
        |                                                        |
        |                                                        +--> solenoid safety gate
        |
        +--> bounded latest-preview slot --> drawing/JPEG worker --> /video_feed/{camera}
```

`VideoWorker` owns capture, inference, and preview threads. It passes native-resolution frames to Ultralytics; only the browser preview is resized. The preview queue keeps at most the newest frame per camera so a slow browser cannot accumulate work or delay inference.

Camera and aggregate frame timestamps use the monotonic process clock. Container health and the solenoid controller require every configured camera to have a recent frame, preventing an HTTP-responsive but stalled inference process from being treated as ready.

## Application lifecycle and API

FastAPI lifespan startup starts the video worker and, when explicitly enabled, the solenoid controller. Shutdown stops the solenoid controller first and then the video worker. A settings update is validated, stops the actuator loop, restarts video with a complete new settings object, and starts the actuator loop only after the video restart succeeds.

The current routes are:

| Method and path | Behavior |
| --- | --- |
| `GET /` | Browser counter UI |
| `GET /video_feed` | Camera 0 MJPEG stream |
| `GET /video_feed/{camera_index}` | Selected camera MJPEG stream |
| `GET /stats` | Aggregate/camera inference and solenoid state |
| `POST /settings` | Validated partial settings update followed by a worker restart |
| `POST /stop` | Stops the solenoid and video worker |

The API currently has no authentication or authorization. It must remain on a trusted, segmented network until the controls described in [production readiness](production-readiness.md) are implemented.

## Counting rules

Ultralytics results are normalized into `CountSummary` values. Overlapping OBB detections are clustered using intersection area relative to the smaller box. Within a cluster, a defect classification takes precedence over a good classification; otherwise the highest-confidence candidate wins. This prevents duplicate boxes from being counted twice and makes defect handling conservative.

## Actuator safety invariants

The solenoid feature defaults to disabled. When enabled:

- Intake and discharge channels must be distinct and within the selected expander's range.
- Outputs are initialized to their inactive electrical level before pins become outputs.
- Discharge is allowed only when every configured camera was running, produced a recent frame, and completed inference during the current inspection window.
- Stop, inference loss, and controller exceptions force both configured outputs inactive.
- Camera settings restarts stop the controller before touching the video worker.

These software checks reduce risk but are not a substitute for an independent hardware watchdog, interlock, emergency stop, or correctly rated valve driver.

## Dataset and model lifecycle

Run dataset preparation from the repository root. It validates image/label pairing, class IDs, OBB row shape, finite coordinates, and nonzero polygon area, then creates deterministic splits using the configured seed:

```bash
.venv/bin/python -m src.capsule_yolo.prepare_dataset
```

`configs/data/capsule.yaml` uses the portable repository-relative `data/prepared` path. Dataset preparation rewrites the same portable path when the output is inside this repository and uses an absolute path only for an explicitly external output directory.

Training writes Ultralytics artifacts under `runs/`; the selected best checkpoint is copied to `models/trained/`. ONNX and TensorRT exports use FP16 by default. TensorRT engines must be built on a compatible target because they are coupled to the CUDA, TensorRT, GPU, and model input shape.

## Configuration precedence

Runtime defaults live in `CounterSettings` and `SolenoidSettings`. Environment variables initialize those values at process import. A validated `POST /settings` request can change video/inference settings for the running process, but it does not persist them to disk or change solenoid wiring/timing configuration.

Deployment environment examples are in `.env.jetson.example` and `compose.jetson.yml`. Treat I2C bus numbers, device nodes, group IDs, camera overlays, and base-image tags as device-specific values that must be verified on each Jetson.

## Dependency model

`requirements.shared.txt` is the exact application-level lock consumed by both local Python and the Jetson Docker build. `scripts/verify_dependency_versions.py` fails local tests or the image build when an installed shared version drifts from that file.

`requirements.txt` adds exact local/desktop builds of PyTorch, torchvision, and OpenCV. Docker deliberately does not use those three pins: Jetson PyTorch/torchvision come from the L4T-compatible NVIDIA base image, and OpenCV comes from the distribution package so CSI capture retains GStreamer support. The image constrains pip against its existing NVIDIA torch versions, removes any transitive generic OpenCV wheel, and then verifies the shared lock.

## Local verification

From an activated virtual environment:

```bash
scripts/run_tests.sh
```

The wrapper uses `.venv/bin/python` without requiring activation and supplies the bundled cuDSS library path when this local CUDA wheel needs it. Select another environment explicitly with:

```bash
CAPSULE_TEST_PYTHON=/path/to/python scripts/run_tests.sh
```

The unit suite is intentionally hardware-independent. Camera, Argus, autofocus, I2C, GPIO polarity, and full container behavior still require target-device acceptance tests.
