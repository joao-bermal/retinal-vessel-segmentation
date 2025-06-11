import os
import cv2
import torch
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/unet_drive.pth"
IMAGE_SIZE = (608, 576)

input_dir = 'data/drive/test/images'
output_dir = 'predicted_masks/drive_test'
os.makedirs(output_dir, exist_ok=True)

# Carrega modelo treinado
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Processamento
for filename in os.listdir(input_dir):
    if not filename.endswith('.tif'):
        continue

    print(f"🖼️ Processando: {filename}")

    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Imagem não carregada: {img_path}")
        continue

    # === Pré-processamento ===
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image[:, :, 1] = clahe.apply(green)

    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_tensor = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        pred = prob.squeeze().cpu().numpy()

    print(f"{filename}: min={pred.min():.4f}, max={pred.max():.4f}")

    # === Plot e visualização ===
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    # Probabilidade (raw)
    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='gray')
    plt.title("Predição (raw)")
    plt.axis('off')

    # Máscara binária
    bin_mask = (pred > 0.5).astype(np.uint8) * 255
    plt.subplot(1, 3, 3)
    plt.imshow(bin_mask, cmap='gray')
    plt.title("Predição (>0.5)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()  # <-- ESSENCIAL para exibir os plots!

    # Salva predição binária
    save_path = os.path.join(output_dir, filename.replace('.tif', '.png'))
    cv2.imwrite(save_path, bin_mask)
