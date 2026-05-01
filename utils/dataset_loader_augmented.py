import random
import cv2
import torch
from torchvision import transforms
from utils.dataset_loader import MultiFeatureImageDataset

class AugmentedMultiFeatureImageDataset(MultiFeatureImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(3)
            ], p=0.3),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        full_img, faces, freq, color, edge, texture, label = super().__getitem__(idx)

        if random.random() < 0.5:
            full_img = self.aug((full_img * 255).byte().permute(1,2,0).numpy())

        return full_img, faces, freq, color, edge, texture, label
