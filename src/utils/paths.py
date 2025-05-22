from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
