import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

# Silence OpenCV warnings
try:
    cv2.setLogLevel(0)
except:
    pass


class MultiFeatureImageDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_root,
        face_root,
        freq_root,
        color_root,
        edge_root,
        texture_root,
        split,                # train / val / test
        image_size=224
    ):
        self.df = pd.read_csv(csv_file)

        self.image_root = image_root
        self.face_root = face_root
        self.freq_root = freq_root
        self.color_root = color_root
        self.edge_root = edge_root
        self.texture_root = texture_root

        self.split = split
        self.image_size = image_size

        # ---------------- RGB TRANSFORMS ----------------
        # Train-time augmentation
        self.rgb_tf_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)],
                p=0.2
            ),
            transforms.ToTensor()
        ])

        # Validation / Test (NO augmentation)
        self.rgb_tf_eval = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # ---------------- GRAYSCALE TRANSFORM ----------------
        self.gray_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # ---------------- CSV VALIDATION ----------------
        if "filename" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                f"CSV must contain ['filename', 'label'] columns. Found: {self.df.columns}"
            )

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_gray(self, path):
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return self.gray_tf(img)
        return torch.zeros((1, self.image_size, self.image_size))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["filename"]
        label = int(row["label"])  # 0 = real, 1 = fake
        cls = "real" if label == 0 else "fake"

        if not isinstance(filename, str) or filename.strip() == "":
            raise ValueError(f"Invalid filename at index {idx}: {row.to_dict()}")

        label = torch.tensor(label, dtype=torch.float32)

        # ---------------- FULL IMAGE ----------------
        img_path = os.path.join(
            self.image_root,
            self.split,
            cls,
            filename
        )

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.split == "train":
            full_img = self.rgb_tf_train(img)
        else:
            full_img = self.rgb_tf_eval(img)

        img_id = os.path.splitext(filename)[0]

        # ---------------- FACE IMAGES ----------------
        face_dir = os.path.join(
            self.face_root,
            self.split,
            cls,
            img_id
        )

        face_tensors = []
        if os.path.exists(face_dir):
            for f in os.listdir(face_dir):
                face = cv2.imread(os.path.join(face_dir, f))
                if face is not None:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    if self.split == "train":
                        face_tensors.append(self.rgb_tf_train(face))
                    else:
                        face_tensors.append(self.rgb_tf_eval(face))

        # fallback → use full image
        if len(face_tensors) == 0:
            faces_tensor = full_img
        else:
            faces_tensor = torch.mean(
                torch.stack(face_tensors), dim=0
            )

        # ---------------- FORENSIC FEATURES ----------------
        freq = self._load_gray(
            os.path.join(self.freq_root, self.split, cls, filename)
        )
        color = self._load_gray(
            os.path.join(self.color_root, self.split, cls, filename)
        )
        edge = self._load_gray(
            os.path.join(self.edge_root, self.split, cls, filename)
        )
        texture = self._load_gray(
            os.path.join(self.texture_root, self.split, cls, filename)
        )

        return (
            full_img,        # (3,H,W)
            faces_tensor,    # (3,H,W)
            freq,            # (1,H,W)
            color,           # (1,H,W)
            edge,            # (1,H,W)
            texture,         # (1,H,W)
            label
        )
