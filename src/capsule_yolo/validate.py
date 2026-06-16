from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import DATA_YAML, DEFAULT_TRAINED_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained capsule counter.")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL), help="Model checkpoint path.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device such as 0, cpu, cuda:0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
