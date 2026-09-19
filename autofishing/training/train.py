"""Local YOLO training entry."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Play Together fishing")
    parser.add_argument("--data", type=Path, default=Path("datasets/data.yaml"))
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="playtogether")
    parser.add_argument("--output", type=Path, default=Path("models/best.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    print(
        f"[train] data={args.data} model={args.model} epochs={args.epochs} "
        f"imgsz={args.imgsz} batch={args.batch} device={args.device}"
    )
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs",
        name=args.name,
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found at {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"[train] copied {best} -> {args.output}")


if __name__ == "__main__":
    main()
