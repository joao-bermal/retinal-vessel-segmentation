import os
import cv2
import torch
import numpy as np
from unet import UNet
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/unet_drive_vessels.pth"
IMAGE_SIZE = (512, 512)

input_dir = 'drive_converted/test/images'
mask_dir = 'drive_converted/test/masks'
output_dir = 'predicted_masks/drive_test'
os.makedirs(output_dir, exist_ok=True)

model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Para armazenar métricas globais
all_accuracies = []
all_ious = []
all_f1s = []

for filename in os.listdir(input_dir):
    if not filename.endswith('.png'):
        continue

    img_path = os.path.join(input_dir, filename)
    img_num = filename.split('_')[0]
    mask_path = os.path.join(mask_dir, f"{img_num}.png")

    if not os.path.exists(mask_path):
        print(f"⚠️ Máscara {mask_path} não encontrada.")
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    # Pré-processamento
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    preprocessed = cv2.merge([blurred, blurred, blurred])

    resized_img = cv2.resize(preprocessed, IMAGE_SIZE).astype('float32') / 255.0
    tensor = torch.tensor(resized_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = output.squeeze().cpu().numpy()
        bin_mask = (pred > 0.5).astype('uint8')

    # Ground truth
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_resized = cv2.resize(gt, IMAGE_SIZE)
    gt_bin = (gt_resized > 127).astype('uint8')

    # Salvar predição
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, bin_mask * 255)

    # Métricas
    y_true = gt_bin.flatten()
    y_pred = bin_mask.flatten()

    acc = accuracy_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    all_accuracies.append(acc)
    all_ious.append(iou)
    all_f1s.append(f1)

# Média das métricas
mean_acc = np.mean(all_accuracies)
mean_iou = np.mean(all_ious)
mean_f1 = np.mean(all_f1s)

print(f"\n✅ Resultado Final nas Imagens de Teste:")
print(f"Acurácia média: {mean_acc:.4f}")
print(f"IoU média: {mean_iou:.4f}")
print(f"F1-score média: {mean_f1:.4f}")
