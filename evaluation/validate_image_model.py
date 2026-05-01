import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

RUN_NAME = datetime.now().strftime("img_%Y%m%d_%H%M%S")
OUT_DIR = f"evaluation/images/{RUN_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)


from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

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

IMAGE_ROOT = "data/processed/images/full_frame"
FACE_ROOT = "data/processed/images/faces"
FREQ_ROOT = "data/processed/images/frequency"
COLOR_ROOT = "data/processed/images/color"
EDGE_ROOT = "data/processed/images/edges"
TEXTURE_ROOT = "data/processed/images/texture"

CHECKPOINT = "checkpoints/images_hard/image_model_hard_epoch_3.pth"# ✅ CORRECT
OUT_DIR = "evaluation/images"

BATCH_SIZE = 16
THRESHOLD = 0.44  # will tune later

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# DATASET (TEST SET)
# --------------------------------------------------
dataset = MultiFeatureImageDataset(
    csv_file=CSV_FILE,
    image_root=IMAGE_ROOT,
    face_root=FACE_ROOT,
    freq_root=FREQ_ROOT,
    color_root=COLOR_ROOT,
    edge_root=EDGE_ROOT,
    texture_root=TEXTURE_ROOT,
    split="test",   # ✅ FIXED
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"📦 Test samples: {len(dataset)}")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = DeepfakeFusionModel().to(DEVICE)

ckpt = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(ckpt["model_state"])
model.eval()

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in loader:
        full_img, faces, freq, color, edge, texture, labels = batch

        full_img = full_img.to(DEVICE)
        faces = faces.to(DEVICE)
        freq = freq.to(DEVICE)
        color = color.to(DEVICE)
        edge = edge.to(DEVICE)
        texture = texture.to(DEVICE)

        logits = model(full_img, faces, freq, color, edge, texture)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy().ravel())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

preds = (all_probs >= THRESHOLD).astype(int)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
acc = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds)
rec = recall_score(all_labels, preds)
f1 = f1_score(all_labels, preds)
roc = roc_auc_score(all_labels, all_probs)

metrics_text = f"""
Accuracy  : {acc:.4f}
Precision : {prec:.4f}
Recall    : {rec:.4f}
F1-score  : {f1:.4f}
ROC-AUC   : {roc:.4f}
"""

print(metrics_text)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(metrics_text)

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------
pd.DataFrame({
    "label": all_labels,
    "probability": all_probs,
    "prediction": preds
}).to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(all_labels, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
disp.plot(cmap="Blues")
plt.title("Image Model Confusion Matrix (Test Set)")
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Image Model ROC Curve (Test Set)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
plt.close()

print("✅ IMAGE TEST EVALUATION COMPLETED")
print(f"📁 Results saved in: {OUT_DIR}")
