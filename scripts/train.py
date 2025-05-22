import sys
from pathlib import Path

# if you need to import from src/… make sure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from src.training.train import train

def main():
    parser = argparse.ArgumentParser(
        description="Train the facial expression YOLO model (assumes data already downloaded)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "9 Facial Expressions you need",
        help="Path to extracted dataset (should contain train/, valid/, data.yaml)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Initial weights (yolo11n.pt)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="facial_expr_yolo11",
        help="Name for this training run"
    )
    args = parser.parse_args()

    data_yaml = args.data_dir / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"[ERROR] data.yaml not found in {args.data_dir}")

    print(f"[INFO] Training with dataset at: {args.data_dir}")
    best_model = train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        weights=args.weights,
        run_name=args.run_name
    )
    print(f"[SUCCESS] Training finished. Best model saved at: {best_model}")

if __name__ == "__main__":
    main()
