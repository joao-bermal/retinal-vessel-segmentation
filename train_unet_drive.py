import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

from dataset import RetinalDataset
from unet import UNet
from losses import dice_loss

# Configurações
batch_size = 4
epochs = 150
learning_rate = 5e-4
model_path = 'models/unet_drive.pth'
os.makedirs("figures", exist_ok=True)

# Dados
images_dir = 'data/drive/training/images'
masks_dir = 'data/drive/training/1st_manual'
image_paths = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.gif')))

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

train_set = RetinalDataset(train_imgs, train_masks, augment=True)
val_set   = RetinalDataset(val_imgs, val_masks, augment=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
bce = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Histórico
history = {
    "dice": [], "accuracy": [], "precision": [],
    "recall": [], "f1": [], "iou": []
}

# Funções auxiliares
def compute_metrics(preds, masks):
    preds = torch.sigmoid(preds)
    preds_bin = (preds > 0.5).float()
    preds_np = preds_bin.cpu().numpy().flatten()
    masks_np = masks.cpu().numpy().flatten()

    intersection = np.sum(preds_np * masks_np)
    union = np.sum((preds_np + masks_np) > 0)

    accuracy = accuracy_score(masks_np, preds_np)
    precision = precision_score(masks_np, preds_np, zero_division=0)
    recall = recall_score(masks_np, preds_np, zero_division=0)
    f1 = f1_score(masks_np, preds_np, zero_division=0)
    dice = 2. * intersection / (np.sum(preds_np) + np.sum(masks_np) + 1e-8)
    iou = intersection / (union + 1e-8)

    return accuracy, precision, recall, f1, dice, iou

def evaluate(model, loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            metrics = compute_metrics(preds, masks)
            all_metrics.append(metrics)

    return np.mean(all_metrics, axis=0)

# Treinamento
print("🏋️ Iniciando treinamento da U-Net...")
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

    acc, prec, rec, f1, dice, iou = evaluate(model, val_loader)

    history["accuracy"].append(acc)
    history["precision"].append(prec)
    history["recall"].append(rec)
    history["f1"].append(f1)
    history["dice"].append(dice)
    history["iou"].append(iou)

    print(f"📘 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}")

    if (epoch + 1) % 10 == 0:
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/unet_epoch{epoch+1}.pth')

# Salvar modelo final 
torch.save(model.state_dict(), model_path)
print(f"✅ Modelo salvo em: {model_path}")

# Gerar gráficos
for metric, values in history.items():
    plt.figure()
    plt.plot(range(1, epochs+1), values, label=metric)
    plt.xlabel('Época')
    plt.ylabel(metric.title())
    plt.title(f'Evolução de {metric.title()} na Validação')
    plt.grid()
    plt.savefig(f'figures/{metric}_curve.png')
    plt.close()
