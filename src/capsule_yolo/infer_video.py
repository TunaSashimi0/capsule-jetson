from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .config import DEFAULT_DEVICE, DEFAULT_TRAINED_MODEL
from .counting import summarize_result
from .drawing import annotated_frame
from .video_source import open_video_capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-time capsule counting on video or camera input.")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL), help="Model checkpoint path.")
    parser.add_argument("--source", default="0", help="Camera index, csi:0/cam0, video file, or stream URL.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device such as 0, cpu, cuda:0.")
    parser.add_argument("--hide-window", action="store_true", help="Run without opening a preview window.")
    parser.add_argument("--output", default=None, help="Optional annotated video output path.")
    return parser


def create_writer(output: str | None, capture: cv2.VideoCapture) -> cv2.VideoWriter | None:
    if not output:
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output, fourcc, fps, (width, height))


def main() -> None:
    args = build_parser().parse_args()
    # TensorRT engine filenames do not encode the Ultralytics task, so declare
    # OBB explicitly instead of letting engine loading fall back to detection.
    model = YOLO(args.model, task="obb")
    capture, source_spec = open_video_capture(args.source)
    source_label = source_spec.label or str(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = create_writer(args.output, capture)
    last_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            summary = summarize_result(result)

            now = time.perf_counter()
            elapsed = max(now - last_time, 1e-6)
            fps = (0.85 * fps) + (0.15 * (1.0 / elapsed)) if fps else 1.0 / elapsed
            last_time = now

            frame_out = annotated_frame(result, summary, fps=fps, conf=args.conf, source_label=source_label)
            if writer is not None:
                writer.write(frame_out)

            print(f"capsules={summary.capsule_count} fps={fps:.1f}", end="\r")

            if not args.hide_window:
                cv2.imshow("YOLO Capsule Counter", frame_out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.hide_window:
            cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
