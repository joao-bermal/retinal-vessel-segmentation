import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformações de treino
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomGamma(p=0.4),
    A.GridDistortion(p=0.3),
    A.OpticalDistortion(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.Resize(512, 512),  # Garante tamanho fixo
    ToTensorV2()
])

class RetinalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, size=(512, 512)):
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_num = img_name.split('_')[0].split('.')[0]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, f"{img_num}.png")

        img = cv2.imread(img_path)
        green = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        processed_img = cv2.merge([blurred, blurred, blurred])

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 15).astype('uint8') * 255

        # === Augmentation
        if self.transform:
            augmented = self.transform(image=processed_img, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0)
        else:
            processed_img = processed_img.astype('float32') / 255.0
            image_tensor = torch.tensor(processed_img.transpose(2, 0, 1), dtype=torch.float32)
            mask = mask.astype('float32') / 255.0
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image_tensor, mask_tensor
