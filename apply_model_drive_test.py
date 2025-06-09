import os
import cv2
import torch
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt

# ==== Configurações ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/unet_drive_vessels.pth"
IMAGE_SIZE = (512, 512)

input_dir = 'drive_converted/test/images'
mask_dir = 'drive_converted/test/masks'
output_dir = 'predicted_masks/drive_test'
os.makedirs(output_dir, exist_ok=True)

# ==== Carrega modelo ====
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== Loop nas imagens ====
for filename in os.listdir(input_dir):
    if not filename.endswith('.png'):
        continue

    img_path = os.path.join(input_dir, filename)
    img_num = filename.split('_')[0]
    mask_path = os.path.join(mask_dir, f"{img_num}.png")

    if not os.path.exists(mask_path):
        print(f"⚠️ Máscara {mask_path} não encontrada.")
        continue

    # === Imagem ===
    original_img = cv2.imread(img_path)
    if original_img is None:
        continue

    # Pré-processamento (CLAHE + Gaussian Blur no canal verde)
    green = original_img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    preprocessed_img = cv2.merge([blurred, blurred, blurred])

    # Redimensionar
    resized_img = cv2.resize(preprocessed_img, IMAGE_SIZE).astype('float32') / 255.0
    tensor = torch.tensor(resized_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # === Predição ===
    with torch.no_grad():
        output = model(tensor)
        pred = output.squeeze().cpu().numpy()
        bin_mask = (pred > 0.6).astype('uint8') * 255

    # === Máscara real ===
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_resized = cv2.resize(gt, IMAGE_SIZE)

    # === Salvar predição ===
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, bin_mask)

    # === Visualização ===
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.title("Imagem")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt_resized, cmap='gray')
    plt.title("Máscara Real")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(bin_mask, cmap='gray')
    plt.title("Predição")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
