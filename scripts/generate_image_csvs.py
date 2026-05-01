import os
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("data")
IMAGE_ROOT = DATA_ROOT / "raw" / "images"
OUT_DIR = DATA_ROOT / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate(split):
    rows = []

    for label_name, label in [("Real", 0), ("Fake", 1)]:
        img_dir = IMAGE_ROOT / split / label_name
        if not img_dir.exists():
            continue

        for img in img_dir.glob("*.jpg"):
            rows.append({
                "filename": img.name,
                "label": label
            })

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / f"{split}_images_clean.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ {out_file} | {len(df)} images")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        generate(split)
