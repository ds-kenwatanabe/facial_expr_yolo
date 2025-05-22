from ultralytics import YOLO
from pathlib import Path

def predict_image(model_path: Path, image_path: Path, conf: float = 0.5, save: bool = True):
    model = YOLO(str(model_path))
    res = model.predict(source=str(image_path), conf=conf, save=save)
    names = [model.names[int(c)] for c in res[0].boxes.cls]
    print(f"Expressions: {names}")
    return res
