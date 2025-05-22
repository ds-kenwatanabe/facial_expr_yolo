from ultralytics import YOLO
from pathlib import Path
from src.utils import paths

def train(
    data_yaml: Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    weights: str = "yolov11n.pt",
    run_name: str = "facial_expr_yolo11"
):
    paths.ensure_dirs()
    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=str(paths.RUNS_DIR),
    )
    best = paths.RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
    print(f"[INFO] Training complete. Best model: {best}")
    return best
