import os
import cv2
import torch
import numpy as np
from PIL import Image
from unet import UNet
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/unet_drive.pth"
IMAGE_SIZE = (608, 576)
input_dir = 'data/drive/training/images'
mask_dir = 'data/drive/training/1st_manual'
roi_dir = 'data/drive/training/mask'
threshold = 0.5

# Carrega modelo
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Métricas
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2.0 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum((y_true + y_pred) > 0)
    return intersection / (union + 1e-8)

all_preds = []
all_targets = []

for filename in tqdm(os.listdir(input_dir)):
    if not filename.endswith('.tif'):
        continue

    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # CLAHE no canal verde
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image[:, :, 1] = clahe.apply(green)

    image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
    tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # === Corrige leitura dos arquivos .gif ===
    mask_path = os.path.join(mask_dir, filename.replace('_training.tif', '_manual1.gif'))
    roi_path = os.path.join(roi_dir, filename.replace('.tif', '_mask.gif'))

    if not os.path.exists(mask_path) or not os.path.exists(roi_path):
        print(f"⚠️ Arquivos ausentes para {filename}")
        continue

    mask = np.array(Image.open(mask_path).convert("L"))
    roi_mask = np.array(Image.open(roi_path).convert("L"))

    mask = cv2.resize(mask, IMAGE_SIZE)
    roi_mask = cv2.resize(roi_mask, IMAGE_SIZE)

    mask = (mask > 127).astype(np.uint8)
    roi_mask = (roi_mask > 127).astype(np.uint8)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output)
        pred = prob.squeeze().cpu().numpy()
        pred_bin = (pred > threshold).astype(np.uint8)

    # Aplica ROI
    mask = mask * roi_mask
    pred_bin = pred_bin * roi_mask

    all_preds.extend(pred_bin.flatten())
    all_targets.extend(mask.flatten())

# === Cálculo de métricas ===
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

if len(all_targets) == 0:
    print("❌ Nenhuma imagem avaliada com sucesso.")
    exit()

tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
dice = dice_score(all_targets, all_preds)
iou = iou_score(all_targets, all_preds)

print("\n📊 Avaliação no conjunto de *treinamento* (com ROI):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"Dice     : {dice:.4f}")
print(f"IoU      : {iou:.4f}")
