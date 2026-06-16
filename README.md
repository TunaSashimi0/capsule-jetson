# YOLO Capsule Counter

This project trains and runs a rudimentary YOLO11n capsule counter for edge-device testing. The current uploaded dataset contains one class, `capsule`, so this implementation counts capsules only. Defect detection should be added later by relabeling data with defect-aware classes such as `capsule_good` and `capsule_defect`.

## Current Dataset

The source labels are expected at:

```text
labeled_data/
  images/
  labels/
  classes.txt
  notes.json
```

The current data audit found 19 images, 19 label files, and 45 labeled capsule boxes.

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
python -m src.capsule_yolo.train --epochs 100 --imgsz 640 --batch 4
```

The trained weights will be written under:

```text
runs/train/capsule_yolo11n/weights/
```

## Validate

```powershell
python -m src.capsule_yolo.validate --model runs/train/capsule_yolo11n/weights/best.pt
```

## Run Video Counter

Use a webcam:

```powershell
python -m src.capsule_yolo.infer_video --model runs/train/capsule_yolo11n/weights/best.pt --source 0
```

Use a video file:

```powershell
python -m src.capsule_yolo.infer_video --model runs/train/capsule_yolo11n/weights/best.pt --source data/samples/test_video.mp4
```

## Run Basic UI

```powershell
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

The UI streams annotated video and shows live capsule count, FPS, model path, confidence threshold, and source.

## Future Defect Detection

When defect examples are available, relabel the dataset with two classes:

```yaml
names:
  0: capsule_good
  1: capsule_defect
```

Then retrain and update the UI counters to show good and defect counts separately.
