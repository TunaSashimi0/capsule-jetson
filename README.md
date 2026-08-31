# YOLO Capsule Counter

This project trains and runs a YOLO11-OBB capsule detector for edge-device testing. The current dataset has two oriented-bounding-box classes: `capsule_defect` (class 0) and `capsule_good` (class 1).

## Current Dataset

The source labels are expected at:

```text
labeled_data/
  images/
  labels/
  classes.txt
  notes.json
```

The current OBB data audit found 325 images, 325 matching label files, and 460 labeled capsule boxes (235 defect and 225 good).

## Environment

Python 3.10 or 3.11 is recommended for the ML stack. A fresh machine needs Git,
Git LFS, Python with `venv` support, and the OpenGL/GLib runtime libraries used
by OpenCV. On Ubuntu/Debian, install them with:

```bash
sudo apt update
sudo apt install -y git git-lfs python3 python3-venv python3-pip libgl1 libglib2.0-0
git lfs install
```

Clone the repository and fetch the LFS-backed source images:

```bash
git clone <repository-url> capsule-jetson
cd capsule-jetson
git lfs pull
```

Create and activate a virtual environment on Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists every direct Python runtime dependency with bounded
versions. PyTorch wheels are platform-specific; use the Jetson container path
below instead of replacing NVIDIA's CUDA-enabled PyTorch with a generic wheel.

## Repository Data and Models

The canonical `labeled_data/` dataset is versioned because it is required to
rebuild the prepared splits and train on a fresh checkout. Its 157 source images
(about 531 MiB total) are stored through Git LFS; labels, class names, and
metadata are ordinary Git files. Run `git lfs pull` after cloning.

When publishing or refreshing the dataset, install Git LFS before staging it,
then verify that the images are represented as LFS objects:

```bash
git lfs install
git add .gitattributes
git add --renormalize labeled_data/images
git add labeled_data
git lfs status
```

The following are reproducible or machine-specific and are intentionally not
committed:

- `data/prepared/`, which is regenerated from `labeled_data/`.
- `datasets/`, which contains datasets downloaded automatically by Ultralytics.
- `runs/`, which contains training and validation output.
- PyTorch checkpoints and ONNX, TensorRT, TorchScript, TFLite, MNN, and similar
  model exports.

Ultralytics downloads the requested base checkpoint (for example,
`yolo11s-obb.pt`) on the first training run. Trained weights stay local under
`models/trained/`; store deployable weights in release or artifact storage if
they need to be shared between machines.

## Prepare Dataset

```powershell
python -m src.capsule_yolo.prepare_dataset
```

This creates:

```text
data/prepared/images/train
data/prepared/images/val
data/prepared/images/test
data/prepared/labels/train
data/prepared/labels/val
data/prepared/labels/test
```

## Train

```powershell
python -m src.capsule_yolo.train --model yolo11s-obb.pt --epochs 100 --imgsz 1280 --batch 4 --workers 0 --device 0 --name capsule_yolo11s_obb --output-model models/trained/capsule_yolo11s_obb_best.pt
```

CUDA training uses automatic mixed precision by default (`--amp`), retaining
FP32 master weights while running eligible operations in FP16. Use `--no-amp`
only for precision troubleshooting.

The YOLO run artifacts will be written under:

```text
runs/train/capsule_yolo11s_obb/weights/
```

After training, `best.pt` is copied to the deployable path tracked through Git
LFS:

```text
models/trained/capsule_yolo11s_obb_best.pt
```

## Validate

```powershell
python -m src.capsule_yolo.validate --model models/trained/capsule_yolo11s_obb_best.pt --device 0
```

## Export FP16

FP16 export is the default. Create a portable FP16 ONNX model with:

```powershell
python -m src.capsule_yolo.export_model --format onnx --device 0
```

Build TensorRT engines on the target NVIDIA device because engine files are
specific to its TensorRT/CUDA stack:

```bash
python -m src.capsule_yolo.export_model --format engine --device 0
```

Use `--no-half` only when an FP32 export is explicitly required.

## Run Video Counter

Use a webcam:

```powershell
python -m src.capsule_yolo.infer_video --model models/trained/capsule_yolo11s_obb_best.pt --source 0 --device 0
```

Use a video file:

```powershell
python -m src.capsule_yolo.infer_video --model models/trained/capsule_yolo11s_obb_best.pt --source data/samples/test_video.mp4 --device 0
```

## Run Basic UI

```powershell
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

The UI streams annotated video and shows live capsule count, FPS, model path, confidence threshold, source, average OBB dimensions, average rotation, and a per-capsule measurement table.

The camera controls expose capture resolution, exposure, analog gain, and ISP
digital gain independently from `CAPSULE_IMGSZ`. For shiny capsules, start at
8000 microseconds with both gains at 1.0, then tune toward mean luma 80–140
while keeping reported clipping below 0.5%. Unsupported USB camera controls may
be ignored by their backend.


## CUDA Smoke Test

For a quick Jetson GPU check without committing a model, run a short 3-epoch smoke test:

```bash
.venv/bin/yolo obb train model=yolo11n-obb.pt data=configs/data/capsule.yaml epochs=3 imgsz=1280 batch=4 workers=0 device=0 project=runs/train name=cuda_smoke_3ep exist_ok=True
.venv/bin/yolo obb predict model=runs/obb/runs/train/cuda_smoke_3ep/weights/best.pt source=data/prepared/images/test imgsz=1280 device=0 save=False
```

The output should report `CUDA:0 (Orin, ...)` and nonzero `GPU_mem` during training.

## Jetson Edge Deployment

This project includes a Jetson-oriented Docker setup for running the FastAPI video counter with NVIDIA's container runtime. Use this path for repeatable edge deployment because the PyTorch/CUDA/OpenCV stack is tightly coupled to the Jetson L4T/JetPack release.

First make sure the current user can access Docker. If `docker info` reports a socket permission error, add the user to the Docker group and start a new login session:

```bash
sudo usermod -aG docker $USER
```

Create an environment file from the example and choose an NVIDIA `l4t-pytorch` base image that matches the device's `/etc/nv_tegra_release` value:

```bash
cp .env.jetson.example .env.jetson
cat /etc/nv_tegra_release
```

Then edit `.env.jetson` and set `JETSON_BASE_IMAGE` to the matching image tag from NVIDIA NGC. This device currently runs JetPack 6.2.2 / Jetson Linux 36.5 (`R36 REVISION 5.0`) with kernel `5.15.185-tegra`, so use an R36.5-compatible image.

Build and run the container with plain Docker:

```bash
scripts/build_jetson_image.sh .env.jetson
scripts/run_jetson_container.sh .env.jetson
```

If Docker Compose v2 is installed, the Compose file is also available:

```bash
docker compose --env-file .env.jetson -f compose.jetson.yml up --build
```

Open the UI from another machine on the same network:

```text
http://<jetson-hostname-or-ip>:8000
```

Useful `.env.jetson` settings:

```text
CAPSULE_MODEL=/app/models/trained/capsule_yolo11s_obb_best.pt
CAPSULE_SOURCE=csi:0
CAPSULE_SECONDARY_SOURCE=csi:1
CAPSULE_DEVICE=0
CAPSULE_IMGSZ=1280
CAPSULE_CAPTURE_WIDTH=3280
CAPSULE_CAPTURE_HEIGHT=2464
CAPSULE_CAPTURE_FPS=21
CAPSULE_EXPOSURE_US=8000
CAPSULE_ANALOG_GAIN=1.0
CAPSULE_DIGITAL_GAIN=1.0
CAPSULE_HALF=true
CAPSULE_PREVIEW_WIDTH=1280
CAPSULE_PREVIEW_FPS=2
CAPSULE_PREVIEW_JPEG_QUALITY=84
CAPSULE_AUTOFOCUS=true
```

Camera source values:

```text
CAPSULE_SOURCE=csi:0   # Jetson CSI / Argus cam0
CAPSULE_SECONDARY_SOURCE=csi:1  # Jetson CSI / Argus cam1
CAPSULE_SOURCE=cam0    # Alias for csi:0
CAPSULE_SOURCE=0       # USB/V4L2 camera at /dev/video0
CAPSULE_SOURCE=gst:<pipeline>  # Custom GStreamer pipeline
```

CSI sources require an OpenCV build with GStreamer enabled. On Jetson, prefer the system `python3-opencv` package or the provided Jetson container path; generic `opencv-python` wheels often do not include GStreamer support. The dual web worker captures each IMX219 independently at 3280×2464, feeds each native frame to the 1280-pixel YOLO model, and scales only the annotated browser preview. It does not stitch the cameras into a lower-resolution inference image. Preview rendering is demand-driven and runs in a separate bounded worker: with no browser connected it does no drawing or JPEG encoding, and when viewed it drops stale frames instead of delaying inference.

### Configure Jetson CSI Camera Hardware

On the Orin Nano Developer Kit, CSI cameras need a matching boot-time device-tree overlay. A Raspberry Pi Camera Module v2 / NoIR-style camera usually uses the IMX219 overlay; the HQ camera usually uses IMX477. The IR/NoIR detail does not change the overlay when the image sensor is still IMX219.

For a single IMX219 camera on CAM0/connector A, run:

```bash
sudo mkdir -p /boot/dtb
sudo cp /boot/tegra234-p3768-0000+p3767-0005-nv.dtb /boot/dtb/
sudo python3 /opt/nvidia/jetson-io/config-by-hardware.py -n 2="Camera IMX219-A"
sudo reboot
```

The `/boot/dtb` copy is needed on this R39.2 image because Jetson-IO searches `/boot/dtb`, while the packaged DTBs are installed directly under `/boot`.

If the camera is plugged into the other CSI connector, use `C` instead of `A`. For an IMX477/HQ camera, use `imx477 A` or `imx477 C`. After reboot, verify Argus can see the camera:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink
```

If that probe reports `No cameras available`, power down and check the ribbon orientation, connector seating, and whether the camera sensor matches the selected overlay.

### Dual Arducam IMX219-AF cameras

This Jetson uses two B0181-style IMX219-AF modules: Argus sensor 0 / CAM A has its focus actuator on I2C bus 10, and sensor 1 / CAM C uses bus 9. Configure the validated dual overlay with:

```bash
scripts/configure_imx219_dual.sh check
sudo scripts/configure_imx219_dual.sh install
sudo reboot
```

After reboot, the web app opens both native 3280×2464 modes and autofocuses each live stream before starting inference. Docker receives `/dev/i2c-9` and `/dev/i2c-10` so the app can send the Arducam 10-bit focus commands while the camera rails are powered. Autofocus status, final position, and the measured sharpness range appear below each view.

To validate both cameras and focus motors independently of the web app:

```bash
sudo scripts/test_imx219_autofocus.py
```

The default `CAPSULE_IMGSZ=1280` matches the trained TensorRT engine. Native capture resolution remains 3280×2464 until YOLO preprocessing. `CAPSULE_PREVIEW_WIDTH`, `CAPSULE_PREVIEW_FPS`, and `CAPSULE_PREVIEW_JPEG_QUALITY` affect only the browser preview. A larger inference size requires a model or TensorRT engine exported for that size.

### MCP23017/MCP23008 solenoid cycle

The web process can control an MCP23017 or MCP23008 through the
[Adafruit MCP230xx CircuitPython library](https://docs.circuitpython.org/projects/mcp230xx/en/latest/).
The controller waits until every configured camera is actively producing inference results, then runs this repeating sequence:

```text
t=0s     channel 0 on; begin the inference inspection window
t=2s     channel 0 off; inference continues
t=30s    channel 1 on, only if every camera inferred during the window
t=33s    channel 1 off; all configured outputs reset inactive
t=153s   120-second cooldown ends and the next cycle may begin
```

If inference stops, a camera stalls, the app stops, or the I2C worker raises an error, both configured channels are forced inactive. Solenoid phase, remaining time, cycle number, and inspection inference count are available in `/stats` and shown in the web header.

The expander detected on this Jetson is an MCP23017-compatible device on `/dev/i2c-7` at address `0x20`. Set the driver board's verified electrical polarity before enabling it:

```text
CAPSULE_SOLENOID_ENABLED=false
CAPSULE_SOLENOID_CHIP=mcp23017
CAPSULE_SOLENOID_I2C_BUS=7
CAPSULE_SOLENOID_I2C_ADDRESS=0x20
CAPSULE_SOLENOID_ACTIVE_HIGH=true
CAPSULE_SOLENOID_INTAKE_CHANNEL=0
CAPSULE_SOLENOID_DISCHARGE_CHANNEL=1
CAPSULE_SOLENOID_INTAKE_SECONDS=2
CAPSULE_SOLENOID_INSPECTION_SECONDS=30
CAPSULE_SOLENOID_DISCHARGE_SECONDS=3
CAPSULE_SOLENOID_COOLDOWN_SECONDS=120
```

Keep `CAPSULE_SOLENOID_ENABLED=false` until `CAPSULE_SOLENOID_ACTIVE_HIGH` is known. Choosing the wrong polarity can energize a valve while the application considers it off.

The default trained OBB model is tracked through Git LFS at `models/trained/capsule_yolo11s_obb_best.pt`. Run `git lfs pull` after cloning to retrieve the model and labeled images, or set `CAPSULE_MODEL` to another model path mounted inside the container.
