from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import DATA_YAML, DEFAULT_DEVICE, DEFAULT_MODEL_IMGSZ, DEFAULT_TRAINED_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained OBB capsule counter.")
    parser.add_argument("--model", default=str(DEFAULT_TRAINED_MODEL), help="Model checkpoint path.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_MODEL_IMGSZ, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device such as 0, cpu, cuda:0.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 CUDA validation by default.",
    )
    parser.add_argument(
        "--task",
        choices=("detect", "obb", "classify", "segment", "pose"),
        default=None,
        help="Optional model task. Exported OBB models may require --task obb.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model, task=args.task) if args.task else YOLO(args.model)
    validation_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
    }
    if args.task:
        validation_args["task"] = args.task
    model.val(**validation_args)


if __name__ == "__main__":
    main()
