import torch
import cv2
import numpy as np
from pathlib import Path

from models.fusion_model import DeepfakeFusionModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/images/image_model_best.pth"

IMG_SIZE = 224


# --------------------------------------------------
# IMAGE PREPROCESS
# --------------------------------------------------

def load_image(path):

    img = cv2.imread(str(path))

    if img is None:
        raise RuntimeError(
            f"Cannot read image: {path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img / 255.0

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

model = DeepfakeFusionModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@torch.no_grad()
def predict_image(image_path):

    image_path = Path(image_path)

    img = load_image(image_path).to(DEVICE)

    # Temporary reuse
    logits = model(
        img,          # full image
        img,          # face
        img[:, :1],   # freq
        img[:, :1],   # color
        img[:, :1],   # edge
        img[:, :1],   # texture
    )

    probability = torch.sigmoid(logits).item()

    label = "FAKE" if probability >= 0.5 else "REAL"

    return label