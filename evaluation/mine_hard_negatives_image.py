import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.dataset_loader import MultiFeatureImageDataset
from models.fusion_model import DeepfakeFusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FILE = "data/splits/val_images_clean.csv"
OUT_CSV = "data/splits/train_images_hard.csv"
CHECKPOINT = "checkpoints/images/image_model_best.pth"

dataset = MultiFeatureImageDataset(
    csv_file=CSV_FILE,
    image_root="data/processed/images/full_frame",
    face_root="data/processed/images/faces",
    freq_root="data/processed/images/frequency",
    color_root="data/processed/images/color",
    edge_root="data/processed/images/edges",
    texture_root="data/processed/images/texture",
    split="val",
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = DeepfakeFusionModel().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

hard_rows = []

with torch.no_grad():
    for idx, batch in enumerate(loader):
        full_img, faces, freq, color, edge, texture, labels = batch

        logits = model(
            full_img.to(DEVICE),
            faces.to(DEVICE),
            freq.to(DEVICE),
            color.to(DEVICE),
            edge.to(DEVICE),
            texture.to(DEVICE),
        )

        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

        for i, p in enumerate(preds):
            if p != labels[i].item():
                hard_rows.append(dataset.df.iloc[idx * 16 + i])

hard_df = pd.DataFrame(hard_rows)
hard_df.to_csv(OUT_CSV, index=False)

print(f"🔥 Hard negatives saved: {len(hard_df)} → {OUT_CSV}")
