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
        self.height, self.width = 576, 608  # tamanho compatível com U-Net e DRIVE

        # Augmentations para treino
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

        # Apenas resize + normalize
        self.base_transform = A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # === Imagem .tif ===
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # CLAHE no canal verde
        green = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image[:, :, 1] = clahe.apply(green)

        # === Máscara .gif ===
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 15).astype(np.uint8) * 255  # binariza
        mask = mask.astype(np.uint8)

        # === Transforms ===
        transform = self.train_transform if self.augment else self.base_transform
        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        # Normaliza máscara para [0, 1] e adiciona canal
        mask = mask.float() / 255.0
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask