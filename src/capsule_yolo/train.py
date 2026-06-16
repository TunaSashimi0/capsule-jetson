from __future__ import annotations

import argparse

from ultralytics import YOLO

from .config import DATA_YAML, DEFAULT_BASE_MODEL, PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO11n-OBB capsule counter.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base OBB model or checkpoint.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. Use 0 on Windows.")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"), help="Output project directory.")
    parser.add_argument("--name", default="capsule_yolo11n_obb", help="Run name.")
    parser.add_argument("--device", default=None, help="Device such as 0, cpu, cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
