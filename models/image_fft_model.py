import torch
import torch.nn as nn
from torchvision import models

class ImageFFTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_cnn = models.efficientnet_b0(pretrained=True)
        self.image_cnn.classifier = nn.Identity()

        self.fft_cnn = models.efficientnet_b0(pretrained=True)
        self.fft_cnn.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, fft):
        img_feat = self.image_cnn(img)
        fft_feat = self.fft_cnn(fft)

        fused = torch.cat((img_feat, fft_feat), dim=1)
        return self.fc(fused)
