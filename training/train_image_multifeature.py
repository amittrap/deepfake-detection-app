import os
import sys
import torch
from pathlib import Path

# --------------------------------------------------
# ADD PROJECT ROOT
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# --------------------------------------------------
# WINDOWS SAFETY
# --------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from utils.dataset_loader import MultiFeatureImageDataset
from models.fusion_model import DeepfakeFusionModel


# --------------------------------------------------
# FOCAL LOSS
# --------------------------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


# --------------------------------------------------
# CHECKPOINT UTILS
# --------------------------------------------------
def save_checkpoint(epoch, model, optimizer, scaler, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"] + 1


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    # ---------------- CONFIG ----------------
    DATA_ROOT = "data"
    CSV_ROOT = os.path.join(DATA_ROOT, "splits")

    IMAGE_ROOT = os.path.join(DATA_ROOT, "processed", "images", "full_frame")
    FACE_ROOT = os.path.join(DATA_ROOT, "processed", "images", "faces")
    FREQ_ROOT = os.path.join(DATA_ROOT, "processed", "images", "frequency")
    COLOR_ROOT = os.path.join(DATA_ROOT, "processed", "images", "color")
    EDGE_ROOT = os.path.join(DATA_ROOT, "processed", "images", "edges")
    TEXTURE_ROOT = os.path.join(DATA_ROOT, "processed", "images", "texture")

    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LR = 3e-4

    CHECKPOINT_DIR = "checkpoints/images"
    RESUME_CKPT = None  # e.g. "checkpoints/images/epoch_1.pth"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", DEVICE)

    # ---------------- DATASET ----------------
    train_dataset = MultiFeatureImageDataset(
        csv_file=os.path.join(CSV_ROOT, "train_images_clean.csv"),
        image_root=IMAGE_ROOT,
        face_root=FACE_ROOT,
        freq_root=FREQ_ROOT,
        color_root=COLOR_ROOT,
        edge_root=EDGE_ROOT,
        texture_root=TEXTURE_ROOT,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    print(f"📦 Training samples: {len(train_dataset)}")

    # ---------------- MODEL ----------------
    model = DeepfakeFusionModel().to(DEVICE)

    # 🔒 FREEZE BACKBONE (WARM-UP)
    for p in model.rgb_face_encoder.features.parameters():
        p.requires_grad = False

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    start_epoch = 0

    # ---------------- RESUME ----------------
    if RESUME_CKPT:
        start_epoch = load_checkpoint(
            RESUME_CKPT, model, optimizer, scaler, DEVICE
        )
        print(f"🔁 Resumed from epoch {start_epoch}")

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(start_epoch, NUM_EPOCHS):

        # 🔓 UNFREEZE AFTER EPOCH 2
        if epoch == 2:
            print("🔓 Unfreezing EfficientNet backbone")
            for p in model.rgb_face_encoder.features.parameters():
                p.requires_grad = True

        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):

            full_img, faces, freq, color, edge, texture, labels = batch

            full_img = full_img.to(DEVICE)
            faces = faces.to(DEVICE)
            freq = freq.to(DEVICE)
            color = color.to(DEVICE)
            edge = edge.to(DEVICE)
            texture = texture.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                logits = model(
                    full_img, faces, freq, color, edge, texture
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if step % 200 == 0:
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} "
                    f"| Step {step}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"[TRAIN] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
        save_checkpoint(epoch, model, optimizer, scaler, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

    print("✅ IMAGE TRAINING COMPLETED")


if __name__ == "__main__":
    main()
