import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# ADD PROJECT ROOT
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.dataset_loader import MultiFeatureImageDataset
from models.fusion_model import DeepfakeFusionModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FILE = "data/splits/test_images_clean.csv"
CHECKPOINT = "checkpoints/images_hard/image_model_hard_epoch_3.pth"
IMAGE_ROOT = "data/processed/images/full_frame"
FACE_ROOT = "data/processed/images/faces"
FREQ_ROOT = "data/processed/images/frequency"
COLOR_ROOT = "data/processed/images/color"
EDGE_ROOT = "data/processed/images/edges"
TEXTURE_ROOT = "data/processed/images/texture"

BATCH_SIZE = 16
OUT_DIR = "evaluation/images_threshold"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
dataset = MultiFeatureImageDataset(
    csv_file=CSV_FILE,
    image_root=IMAGE_ROOT,
    face_root=FACE_ROOT,
    freq_root=FREQ_ROOT,
    color_root=COLOR_ROOT,
    edge_root=EDGE_ROOT,
    texture_root=TEXTURE_ROOT,
    split="test",
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📦 Test samples: {len(dataset)}")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = DeepfakeFusionModel().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# --------------------------------------------------
# INFERENCE (once)
# --------------------------------------------------
all_probs, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        full_img, faces, freq, color, edge, texture, labels = batch

        logits = model(
            full_img.to(DEVICE),
            faces.to(DEVICE),
            freq.to(DEVICE),
            color.to(DEVICE),
            edge.to(DEVICE),
            texture.to(DEVICE),
        )

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().ravel())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# --------------------------------------------------
# THRESHOLD SWEEP
# --------------------------------------------------
results = []

for t in np.arange(0.30, 0.70, 0.02):
    preds = (all_probs >= t).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    results.append({
        "threshold": round(t, 2),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

df = pd.DataFrame(results).sort_values(
    by=["recall", "f1"], ascending=False
)

df.to_csv(os.path.join(OUT_DIR, "threshold_results.csv"), index=False)

print("\n🔥 TOP THRESHOLDS (Max Recall):")
print(df.head(10))

print(f"\n📁 Saved to: {OUT_DIR}")
