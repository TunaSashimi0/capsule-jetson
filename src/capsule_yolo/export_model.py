from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import DEFAULT_TRAINED_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained capsule counter for edge deployment.")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL), help="Model checkpoint path.")
    parser.add_argument("--format", default="onnx", help="Export format, such as onnx or engine.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported.")
    parser.add_argument("--device", default=None, help="Device such as 0, cpu, cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.export(format=args.format, imgsz=args.imgsz, half=args.half, device=args.device)


if __name__ == "__main__":
    main()
