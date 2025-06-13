import os
import cv2
import torch
import numpy as np
from PIL import Image
from unet import UNet
from sklearn.metrics import confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/unet_drive.pth"
IMAGE_SIZE = (608, 576)
input_dir = 'data/drive/training/images'
mask_dir = 'data/drive/training/1st_manual'
roi_dir = 'data/drive/training/mask'
threshold = 0.5
os.makedirs("figures", exist_ok=True)

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

# Acumuladores
all_preds = []
all_targets = []
all_preds_raw = []
dice_list = []
iou_list = []

# Avaliação por imagem
for filename in tqdm(os.listdir(input_dir)):
    if not filename.endswith('.tif'):
        continue

    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image[:, :, 1] = clahe.apply(green)

    image_resized = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
    tensor = torch.tensor(image_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)

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
    pred = pred * roi_mask

    # Acumula resultados
    all_preds.extend(pred_bin.flatten())
    all_targets.extend(mask.flatten())
    all_preds_raw.extend(pred.flatten())

    dice_list.append(dice_score(mask.flatten(), pred_bin.flatten()))
    iou_list.append(iou_score(mask.flatten(), pred_bin.flatten()))

    # Salva comparação visual de alguns exemplos
    if filename.startswith(("21", "23", "36", "40")):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original")
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Máscara Manual")
        axs[2].imshow(pred_bin, cmap='gray')
        axs[2].set_title("Predição U-Net")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"figures/comparison_{filename.replace('.tif','')}.png")
        plt.close()

# Métricas globais
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_preds_raw = np.array(all_preds_raw)

tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
dice = dice_score(all_targets, all_preds)
iou = iou_score(all_targets, all_preds)

# Exibe resultados
print("\n📊 Avaliação no conjunto de *treinamento* (com ROI):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"Dice     : {dice:.4f}")
print(f"IoU      : {iou:.4f}")

# Curva Precision-Recall
precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_preds_raw)
plt.figure()
plt.plot(recall_curve, precision_curve, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.grid()
plt.savefig("figures/pr_curve.png")
plt.close()

# Matriz de Confusão
disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[tn, fp], [fn, tp]]), display_labels=["Fundo", "Vaso"])
disp.plot(cmap='Blues', values_format='.2g')
plt.title("Matriz de Confusão - Segmentação")
plt.savefig("figures/confusion_matrix.png")
plt.close()

# Boxplot Dice / IoU
plt.figure()
sns.boxplot(data=[dice_list, iou_list])
plt.xticks([0, 1], ['Dice', 'IoU'])
plt.title('Distribuição das Métricas por Imagem')
plt.grid()
plt.savefig("figures/metric_distribution.png")
plt.close()
