# YOLO Capsule Counter

This project trains and runs a rudimentary YOLO11n-OBB capsule counter for edge-device testing. The current uploaded dataset contains one class, `capsule`, with oriented bounding box labels, so this implementation counts capsules and reports each capsule's pixel width, pixel height, and image-plane rotation. Defect detection should be added later by relabeling data with defect-aware classes such as `capsule_good` and `capsule_defect`.

## Current Dataset

The source labels are expected at:

```text
labeled_data/
  images/
  labels/
  classes.txt
  notes.json
```

The current OBB data audit found 29 images, 29 label files, and 93 labeled capsule boxes.

## Environment

Python 3.10 or 3.11 is recommended for the ML stack. This machine currently reports Python 3.14, which may not be supported by PyTorch/Ultralytics wheels yet.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

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
python -m src.capsule_yolo.train --epochs 100 --imgsz 640 --batch 4 --device 0
```

The YOLO run artifacts will be written under:

```text
runs/train/capsule_yolo11n_obb/weights/
```

After training, `best.pt` is copied to the deployable, Git-tracked path:

```text
models/trained/capsule_yolo11n_obb_best.pt
```

## Validate

```powershell
python -m src.capsule_yolo.validate --model models/trained/capsule_yolo11n_obb_best.pt --device 0
```

## Run Video Counter

Use a webcam:

```powershell
python -m src.capsule_yolo.infer_video --model models/trained/capsule_yolo11n_obb_best.pt --source 0 --device 0
```

Use a video file:

```powershell
python -m src.capsule_yolo.infer_video --model models/trained/capsule_yolo11n_obb_best.pt --source data/samples/test_video.mp4 --device 0
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


## CUDA Smoke Test

For a quick Jetson GPU check without committing a model, run a short 3-epoch smoke test:

```bash
.venv/bin/yolo obb train model=yolo11n-obb.pt data=configs/data/capsule.yaml epochs=3 imgsz=640 batch=4 workers=0 device=0 project=runs/train name=cuda_smoke_3ep exist_ok=True
.venv/bin/yolo obb predict model=runs/obb/runs/train/cuda_smoke_3ep/weights/best.pt source=data/prepared/images/test imgsz=640 device=0 save=False
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

Then edit `.env.jetson` and set `JETSON_BASE_IMAGE` to the matching image tag from NVIDIA NGC. This device was observed as `R39 REVISION 2.0`, so use an R39.2-compatible `l4t-pytorch` image when available.

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
CAPSULE_MODEL=/app/models/trained/capsule_yolo11n_obb_best.pt
CAPSULE_SOURCE=csi:0
CAPSULE_CAMERA_DEVICE=/dev/video0
CAPSULE_DEVICE=0
```

Camera source values:

```text
CAPSULE_SOURCE=csi:0   # Jetson CSI / Argus cam0
CAPSULE_SOURCE=cam0    # Alias for csi:0
CAPSULE_SOURCE=0       # USB/V4L2 camera at /dev/video0
CAPSULE_SOURCE=gst:<pipeline>  # Custom GStreamer pipeline
```

CSI sources require an OpenCV build with GStreamer enabled. On Jetson, prefer the system `python3-opencv` package or the provided Jetson container path; generic `opencv-python` wheels often do not include GStreamer support. If the camera has an IR mode or IR sensor, it is only useful to this model if it produces a normal image stream and the model has been trained or validated on similar IR imagery.

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

Pi Camera Module 3 / IMX708 note: Raspberry Pi Camera Module 3 uses a Sony IMX708 sensor. This Jetson R39.2 image does not include an `nv_imx708` kernel module or an IMX708 Jetson-IO overlay, so it cannot be configured as a working Argus camera without installing a vendor/kernel driver and matching device-tree overlay. Raspberry Pi's upstream IMX708 overlay uses I2C address `0x1a`; the stock Jetson IMX219 overlays probe address `0x10`, so IMX219 failure logs are not proof that an IMX708 module is bad.

The default trained OBB model is expected at `models/trained/capsule_yolo11n_obb_best.pt`. If that file is missing on a fresh checkout, train with the labeled data or temporarily set `CAPSULE_MODEL` to a model path that exists inside the container.

## Future Defect Detection

When defect examples are available, relabel the dataset with two classes:

```yaml
names:
  0: capsule_good
  1: capsule_defect
```

Then retrain and update the UI counters to show good and defect counts separately.
