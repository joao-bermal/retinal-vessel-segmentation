import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_image(img_path, out_path):
    img = cv2.imread(img_path)
    if img is not None:
        out_path = out_path.replace('.tif', '.png')
        cv2.imwrite(out_path, img)
    else:
        print(f"❌ Erro ao ler imagem: {img_path}")

def convert_mask(mask_path, out_path):
    try:
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 15).astype('uint8') * 255
        out_path = out_path.replace('.gif', '.png')
        cv2.imwrite(out_path, mask)
    except Exception as e:
        print(f"❌ Erro ao converter máscara {mask_path}: {e}")

# Diretórios de entrada
training_images_dir = r"data/training/images"
training_manual_dir = r"data/training/1st_manual"
test_images_dir = r"data/test/images"
test_manual_dir = r"data/test/mask"

# Diretórios de saída
out_train_img_dir = r"drive_converted/train/images"
out_train_mask_dir = r"drive_converted/train/masks"
out_test_img_dir = r"drive_converted/test/images"
out_test_mask_dir = r"drive_converted/test/masks"

for d in [out_train_img_dir, out_train_mask_dir, out_test_img_dir, out_test_mask_dir]:
    os.makedirs(d, exist_ok=True)

print("🔄 Convertendo imagens de treinamento...")
for filename in tqdm(os.listdir(training_images_dir)):
    if filename.endswith(".tif"):
        src = os.path.join(training_images_dir, filename)
        dst = os.path.join(out_train_img_dir, filename.replace(".tif", ".png"))
        convert_image(src, dst)

print("🎯 Convertendo máscaras de vasos (1st_manual)...")
for filename in tqdm(os.listdir(training_manual_dir)):
    if filename.endswith(".gif"):
        src = os.path.join(training_manual_dir, filename)
        num = filename.split('_')[0]  # "21_manual1.gif" → "21"
        dst = os.path.join(out_train_mask_dir, f"{num}.png")
        convert_mask(src, dst)

print("🔄 Convertendo imagens de teste...")
for filename in tqdm(os.listdir(test_images_dir)):
    if filename.endswith(".tif"):
        src = os.path.join(test_images_dir, filename)
        dst = os.path.join(out_test_img_dir, filename.replace(".tif", ".png"))
        convert_image(src, dst)

print("🎯 Convertendo máscaras de vasos de teste...")
for filename in tqdm(os.listdir(test_manual_dir)):
    if filename.endswith("_mask.gif"):  # Apenas as máscaras reais
        src = os.path.join(test_manual_dir, filename)
        num = filename.split('_')[0]  # "01_test_mask.gif" → "01"
        dst = os.path.join(out_test_mask_dir, f"{num}.png")
        convert_mask(src, dst)

print("✅ Conversão finalizada!")
