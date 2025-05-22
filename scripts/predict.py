import argparse, sys
from pathlib import Path
from src.inference.predict import predict_image

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path, help="Image Path")
    p.add_argument("--model", type=Path, default="runs/detect/facial_expr_yolo11/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.5)
    args = p.parse_args()

    if not args.model.exists():
        sys.exit(f"[WARNING] Model {args.model} not found. Train it first with scripts/train.py")
    predict_image(args.model, args.image, conf=args.conf, save=True)
