from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import DEFAULT_DEVICE, DEFAULT_MODEL_IMGSZ, DEFAULT_TRAINED_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained capsule counter for edge deployment.")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL), help="Model checkpoint path.")
    parser.add_argument("--format", default="onnx", help="Export format, such as onnx or engine.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_MODEL_IMGSZ, help="Export image size.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export FP16 by default; use --no-half for FP32.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device such as 0, cpu, cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        quantize=16 if args.half else 32,
        device=args.device,
    )


if __name__ == "__main__":
    main()
