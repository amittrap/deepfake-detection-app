import sys
from pathlib import Path
import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

from utils.video_dataset_loader import MultiFeatureVideoDataset
from models.video_fusion_model import VideoFusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "checkpoints/videos/video_fusion_epoch_10.pth"

dataset = MultiFeatureVideoDataset(
    csv_file="data/splits/test_videos_clean.csv",
    frame_root="data/processed/videos/frames",
    freq_root="data/processed/videos/frequency",
    color_root="data/processed/videos/color",
    edge_root="data/processed/videos/edges",
    texture_root="data/processed/videos/texture"
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = VideoFusionModel().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)["model_state"])
model.eval()

y_true = []
y_scores = []

with torch.no_grad():
    for rgb, freq, color, edge, texture, labels, _ in loader:
        rgb = rgb.to(DEVICE)
        freq = freq.to(DEVICE)
        color = color.to(DEVICE)
        edge = edge.to(DEVICE)
        texture = texture.to(DEVICE)

        probs = torch.sigmoid(
            model(rgb, freq, color, edge, texture)
        ).cpu().numpy()

        y_scores.extend(1 - probs)
        y_true.extend(labels.numpy())

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Frame-level)")
plt.legend()
plt.show()
