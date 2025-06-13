import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import os
from dataset import RetinalDataset, load_drive_paths, load_stare_paths
from unet import UNet
from losses import dice_loss

# Configurações
batch_size = 4
epochs = 150
learning_rate = 5e-4
model_path = 'models/unet_drive_stare.pth'

# Carrega os caminhos dos dois datasets
drive_imgs, drive_masks = load_drive_paths()
stare_imgs, stare_masks = load_stare_paths()

# Junta tudo
all_imgs = drive_imgs + stare_imgs
all_masks = drive_masks + stare_masks

# Divide treino e validação
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    all_imgs, all_masks, test_size=0.2, random_state=42)

# Datasets e loaders
train_set = RetinalDataset(train_imgs, train_masks, augment=True)
val_set   = RetinalDataset(val_imgs, val_masks, augment=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)

# Modelo e otimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
bce = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Validação
def evaluate(model, loader):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = dice_loss(preds, masks)
            dice_scores.append(1 - loss.item())
    return sum(dice_scores) / len(dice_scores)

# Treinamento
print("🏋️ Iniciando treinamento com DRIVE + STARE...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = 0.5 * bce(preds, masks) + 0.5 * dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    val_dice = evaluate(model, val_loader)
    print(f"📘 Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f}")

    if (epoch + 1) % 10 == 0:
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/unet_combined_epoch{epoch+1}.pth')

# Modelo final
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✅ Modelo salvo em: {model_path}")
