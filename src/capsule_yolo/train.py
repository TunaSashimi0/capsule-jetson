from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from .config import (
    DATA_YAML,
    DEFAULT_BASE_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_TRAINED_MODEL,
    PROJECT_ROOT,
    project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO11-OBB capsule detector.")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base OBB model or checkpoint.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. Use 0 on Windows.")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"), help="Output project directory.")
    parser.add_argument("--name", default="capsule_yolo11n_obb", help="Run name.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device such as 0, cpu, cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--output-model",
        help="Optional path where the run's best.pt will be copied for deployment.",
    )
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

    run_best = Path(args.project) / args.name / "weights" / "best.pt"
    if not run_best.is_absolute():
        run_best = PROJECT_ROOT / run_best
    if run_best.exists():
        output_model = project_path(args.output_model) if args.output_model else DEFAULT_TRAINED_MODEL
        output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_best, output_model)
        print(f"Copied deployable model to {output_model}")
    else:
        print(f"Training finished, but best.pt was not found at {run_best}")


if __name__ == "__main__":
    main()
