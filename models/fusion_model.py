import torch
import torch.nn as nn
from torchvision import models


# -----------------------------
# EfficientNet Feature Extractor
# -----------------------------
class EfficientNetEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()

        base = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(1280, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# -----------------------------
# Small CNN for Forensic Maps
# -----------------------------
class ForensicEncoder(nn.Module):
    def __init__(self, in_ch=1, out_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)


# -----------------------------
# FINAL IMAGE FUSION MODEL
# -----------------------------
class DeepfakeFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared backbone for RGB + Face
        self.rgb_face_encoder = EfficientNetEncoder(out_dim=256)

        # Forensic branches
        self.freq_enc = ForensicEncoder(1, 64)
        self.color_enc = ForensicEncoder(1, 64)
        self.edge_enc = ForensicEncoder(1, 64)
        self.texture_enc = ForensicEncoder(1, 64)

        fusion_dim = 256 * 2 + 64 * 4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, full_img, faces, freq, color, edge, texture):
        rgb_feat = self.rgb_face_encoder(full_img)
        face_feat = self.rgb_face_encoder(faces)

        freq_feat = self.freq_enc(freq)
        color_feat = self.color_enc(color)
        edge_feat = self.edge_enc(edge)
        texture_feat = self.texture_enc(texture)

        fused = torch.cat(
            [rgb_feat, face_feat, freq_feat, color_feat, edge_feat, texture_feat],
            dim=1
        )

        return self.classifier(fused).squeeze(1)
