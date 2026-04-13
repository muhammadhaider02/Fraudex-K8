import zipfile
from pathlib import Path
import kaggle

def download():
    assets_dir = Path(__file__).parent
    competition = "ieee-fraud-detection"

    print("Downloading IEEE CIS Fraud Detection dataset...")
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(competition, path=assets_dir, quiet=False)

    zip_path = assets_dir / f"{competition}.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(assets_dir)
        zip_path.unlink()
        print(f"Done. Files saved to {assets_dir}")
    else:
        print("Download failed. Check your kaggle.json and competition access.")

if __name__ == "__main__":
    download()