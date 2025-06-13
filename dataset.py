import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.height, self.width = 576, 608  # compatível com U-Net e avaliação

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(self.height, self.width),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ])

        self.base_transform = A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        green = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image[:, :, 1] = clahe.apply(green)

        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 15).astype(np.uint8) * 255

        transform = self.train_transform if self.augment else self.base_transform
        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].float() / 255.0

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask

def load_drive_paths(base_path="data/drive/training"):
    images = []
    masks = []

    img_dir = os.path.join(base_path, "images")
    mask_dir = os.path.join(base_path, "1st_manual")

    for fname in os.listdir(img_dir):
        if fname.endswith(".tif"):
            img_path = os.path.join(img_dir, fname)
            mask_name = fname.replace("_training.tif", "_manual1.gif")
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                images.append(img_path)
                masks.append(mask_path)
    return images, masks

def load_stare_paths(base_path="data/stare"):
    images = []
    masks = []

    img_dir = os.path.join(base_path, "stare-images")
    mask_dir = os.path.join(base_path, "labels-ah")  # ou labels-vk

    for fname in os.listdir(mask_dir):
        if fname.endswith(".ppm") or fname.endswith(".png"):
            mask_path = os.path.join(mask_dir, fname)
            img_name = fname.replace(".ah.ppm", ".ppm").replace(".vk.ppm", ".ppm")
            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(img_path):
                images.append(img_path)
                masks.append(mask_path)
    return images, masks
