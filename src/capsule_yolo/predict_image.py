from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import DEFAULT_DEVICE, DEFAULT_MODEL_IMGSZ, DEFAULT_TRAINED_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OBB capsule counter predictions on images.")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL), help="Model checkpoint path.")
    parser.add_argument("--source", default="data/prepared/images/test", help="Image, directory, or glob source.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_MODEL_IMGSZ, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device such as 0, cpu, cuda:0.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 CUDA inference by default.",
    )
    parser.add_argument("--save", action="store_true", help="Save annotated images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        save=args.save,
    )


if __name__ == "__main__":
    main()
