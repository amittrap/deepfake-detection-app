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

real_probs = []
fake_probs = []

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

        labels = labels.numpy()

        for p, y in zip(probs, labels):
            if y == 0:
                real_probs.append(p)
            else:
                fake_probs.append(p)

plt.figure()
plt.hist(real_probs, bins=30, alpha=0.6, label="Real frames")
plt.hist(fake_probs, bins=30, alpha=0.6, label="Fake frames")
plt.xlabel("Predicted Probability")
plt.ylabel("Frame Count")
plt.title("Frame-level Prediction Distribution")
plt.legend()
plt.show()
