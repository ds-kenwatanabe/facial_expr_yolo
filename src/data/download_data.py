import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
from src.utils.paths import DATA_DIR, ensure_dirs

DATASET = "aklimarimi/8-facial-expressions-for-yolo"
ZIP_NAME = f"{DATASET.split('/')[-1]}.zip"
EXTRACTED_DIR  = DATA_DIR / "facial_expressions"


def download_and_extract(force: bool = False) -> Path:
    """
    Download the Kaggle dataset via CLI (preferred) or Python API (fallback),
    then extract into DATA_DIR/facial_expressions, flattening the inner folder.
    """
    ensure_dirs()
    zip_path = DATA_DIR / ZIP_NAME

    # 1) Try to locate the Kaggle CLI in your venv
    venv_scripts = Path(sys.executable).parent
    if sys.platform.startswith("win"):
        kaggle_cli = venv_scripts / "kaggle.exe"
    else:
        kaggle_cli = venv_scripts / "kaggle"

    if kaggle_cli.exists():
        # Build command list
        cmd = [
            str(kaggle_cli),
            "datasets", "download",
            "-d", DATASET,
            "-p", str(DATA_DIR)
        ]
        if force:
            cmd.append("--force")

        print(f"[RUNNING] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        # Fallback: Python Kaggle API
        print("[INFO] Kaggle CLI not found in venv; falling back to Python API.")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                DATASET,
                path=str(DATA_DIR),
                quiet=False,
                force=force
            )
        except Exception as e:
            raise RuntimeError(f"Kaggle download failed: {e}")

    # 2) Extract the zip
    if EXTRACTED_DIR.exists() and not force:
        print("[INFO] Dataset already extracted – skipping unzip.")
        return EXTRACTED_DIR

    print(f"[INFO] Extracting {zip_path} → {DATA_DIR} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=str(DATA_DIR))

    # 3) Flatten inner folder named “9 Facial Expressions you need”
    inner = EXTRACTED_DIR / "9 Facial Expressions you need"
    if inner.exists() and inner.is_dir():
        for item in inner.iterdir():
            shutil.move(str(item), str(EXTRACTED_DIR))
        shutil.rmtree(inner)

    print(f"[SUCCESS] Data extracted to: {EXTRACTED_DIR}")
    return EXTRACTED_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download & extract the 9 Facial Expressions YOLO dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files already exist"
    )
    args = parser.parse_args()
    download_and_extract(force=args.force)
