from src.data.download_data import download_and_extract

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-extract the dataset"
    )
    args = parser.parse_args()
    download_and_extract(force=args.force)