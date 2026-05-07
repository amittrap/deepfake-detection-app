import torch
import cv2
import numpy as np
from pathlib import Path

from models.fusion_model import DeepfakeFusionModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# Force CPU for deployment stability
DEVICE = torch.device("cpu")

CHECKPOINT = "checkpoints/images/image_model_best.pth"

# Reduced size for lower RAM usage
IMG_SIZE = 160

# Reduce CPU thread usage
torch.set_num_threads(1)


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

    # Smaller image = faster inference
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32) / 255.0

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    tensor = torch.from_numpy(img).float().unsqueeze(0)

    return tensor


# --------------------------------------------------
# LOAD MODEL ONLY ONCE
# --------------------------------------------------

print("Loading Deepfake Detection Model...")

model = DeepfakeFusionModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

print("Model Loaded Successfully")


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@torch.no_grad()
def predict_image(image_path):

    try:

        image_path = Path(image_path)

        img = load_image(image_path).to(DEVICE)

        # Temporary feature reuse
        logits = model(
            img,          # full image
            img,          # face
            img[:, :1],   # frequency
            img[:, :1],   # color
            img[:, :1],   # edge
            img[:, :1],   # texture
        )

        probability = torch.sigmoid(
            logits
        ).item()

        label = (
            "FAKE"
            if probability >= 0.5
            else "REAL"
        )

        return label

    except Exception as e:

        print(f"Inference Error: {e}")

        return "ERROR"