import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import RetinalDataset, train_transform
from unet import UNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
epochs = 100

images_dir = 'drive_converted/train/images'
masks_dir = 'drive_converted/train/masks'
model_path = 'models/unet_drive_vessels.pth'

dataset = RetinalDataset(images_dir, masks_dir, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("🏋️ Iniciando treinamento da U-Net...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"📘 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✅ Modelo salvo em: {model_path}")
