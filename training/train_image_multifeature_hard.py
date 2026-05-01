import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# --------------------------------------------------
# ADD PROJECT ROOT
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.dataset_loader_augmented import AugmentedMultiFeatureImageDataset
from models.fusion_model import DeepfakeFusionModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FILE = "data/splits/train_images_hard.csv"

IMAGE_ROOT = "data/processed/images/full_frame"
FACE_ROOT = "data/processed/images/faces"
FREQ_ROOT = "data/processed/images/frequency"
COLOR_ROOT = "data/processed/images/color"
EDGE_ROOT = "data/processed/images/edges"
TEXTURE_ROOT = "data/processed/images/texture"

BASE_CHECKPOINT = "checkpoints/images/image_model_best.pth"

# 🔥 NEW SEPARATE FOLDER (NO OVERWRITE)
HARD_CKPT_DIR = "checkpoints/images_hard"
os.makedirs(HARD_CKPT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4

# --------------------------------------------------
# DATASET
# --------------------------------------------------
dataset = AugmentedMultiFeatureImageDataset(
    csv_file=CSV_FILE,
    image_root=IMAGE_ROOT,
    face_root=FACE_ROOT,
    freq_root=FREQ_ROOT,
    color_root=COLOR_ROOT,
    edge_root=EDGE_ROOT,
    texture_root=TEXTURE_ROOT,
    split="train",
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True   # 🔥 IMPORTANT
)

print(f"🔥 Hard samples: {len(dataset)}")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = DeepfakeFusionModel().to(DEVICE)

ckpt = torch.load(BASE_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

# 🔒 Freeze BatchNorm (VERY IMPORTANT)
def freeze_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

model.apply(freeze_batchnorm)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch in loader:
        imgs, faces, freq, color, edge, texture, labels = batch

        imgs = imgs.to(DEVICE)
        faces = faces.to(DEVICE)
        freq = freq.to(DEVICE)
        color = color.to(DEVICE)
        edge = edge.to(DEVICE)
        texture = texture.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            logits = model(imgs, faces, freq, color, edge, texture)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"🔥 Hard Fine-tune Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # ✅ SAVE PER-EPOCH CHECKPOINT (NO OVERWRITE)
    ckpt_path = os.path.join(
        HARD_CKPT_DIR,
        f"image_model_hard_epoch_{epoch+1}.pth"
    )
    torch.save(
        {"model_state": model.state_dict()},
        ckpt_path
    )
    print(f"💾 Saved: {ckpt_path}")

print("✅ HARD NEGATIVE FINE-TUNING COMPLETED")
