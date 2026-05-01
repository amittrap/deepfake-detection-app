import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# ROOTS
# -----------------------------
IMG_ROOT = Path("data/processed/images")
CSV_ROOT = Path("data/splits")

SRC = IMG_ROOT / "full_frame"

OUT_FREQ = IMG_ROOT / "frequency"
OUT_COLOR = IMG_ROOT / "color"
OUT_EDGE = IMG_ROOT / "edges"
OUT_TEXTURE = IMG_ROOT / "texture"

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------
def fft_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def color_map(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def edge_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 80, 150)


def lbp_map(gray):
    h, w = gray.shape
    out = np.zeros((h, w), np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = gray[y, x]
            code = 0
            code |= (gray[y-1, x-1] > c) << 7
            code |= (gray[y-1, x]   > c) << 6
            code |= (gray[y-1, x+1] > c) << 5
            code |= (gray[y,   x+1] > c) << 4
            code |= (gray[y+1, x+1] > c) << 3
            code |= (gray[y+1, x]   > c) << 2
            code |= (gray[y+1, x-1] > c) << 1
            code |= (gray[y,   x-1] > c)
            out[y, x] = code

    return out


# -----------------------------
# MAIN LOOP
# -----------------------------
for split in ["train", "val", "test"]:

    csv_file = CSV_ROOT / f"{split}_images_clean.csv"
    df = pd.read_csv(csv_file)

    print(f"\n🚀 Processing {split.upper()} | {len(df)} images")

    for _, row in df.iterrows():

        filename = row["filename"]
        label = "real" if row["label"] == 0 else "fake"

        src_img = SRC / split / label / filename
        if not src_img.exists():
            raise FileNotFoundError(f"❌ Missing image: {src_img}")

        img = cv2.imread(str(src_img))
        if img is None:
            raise RuntimeError(f"❌ Cannot read image: {src_img}")

        # Create output dirs
        (OUT_FREQ / split / label).mkdir(parents=True, exist_ok=True)
        (OUT_COLOR / split / label).mkdir(parents=True, exist_ok=True)
        (OUT_EDGE / split / label).mkdir(parents=True, exist_ok=True)
        (OUT_TEXTURE / split / label).mkdir(parents=True, exist_ok=True)

        # Write features
        cv2.imwrite(
            str(OUT_FREQ / split / label / filename),
            fft_map(img)
        )

        cv2.imwrite(
            str(OUT_COLOR / split / label / filename),
            color_map(img)
        )

        cv2.imwrite(
            str(OUT_EDGE / split / label / filename),
            edge_map(img)
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(
            str(OUT_TEXTURE / split / label / filename),
            lbp_map(gray)
        )

    print(f"✅ {split.upper()} DONE")

print("\n🎯 ALL IMAGE FORENSIC FEATURES GENERATED (FULLY ALIGNED)")
