import os
import random
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from eval.erfnet import ERFNet
from train.main import MyCoTransform

def uniform_sample(feat_map, stride = 2):
    # feat_map: [C, H, W]
    C, H, W = feat_map.shape
    samples = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            samples.append(feat_map[:, h, w])  # shape: [C]
    return torch.stack(samples)  # shape: [N_samples, C]

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20
image_root = r"D:\semester_3\AML\project\datasets\cityscapes\leftImg8bit\train"  # Update this path

# Initialize transformation and model
co_transform = MyCoTransform(enc=True, augment=False, height=512)
model = ERFNet(num_classes).to(device)
model.eval()

# Collect all image paths
image_paths = []
for city in os.listdir(image_root):
    city_folder = os.path.join(image_root, city)
    for fname in os.listdir(city_folder):
        if fname.endswith("_leftImg8bit.png"):
            image_paths.append(os.path.join(city_folder, fname))

all_features = []

# --- MAIN LOOP ---
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Extracting features"):
        pil_img = Image.open(img_path).convert("RGB")
        dummy_gt = Image.new("L", pil_img.size)  # dummy GT

        input_tensor, _ = co_transform(pil_img, dummy_gt)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        feat_map = model(input_tensor, only_encode=True).squeeze(0).cpu()  # shape: [C, H, W]

        C, H, W = feat_map.shape

        feat_map = uniform_sample(feat_map)

        all_features.append(feat_map)

# --- COMBINE ---
feature_bank = torch.cat(all_features, dim=0)  # [num_images * N, C]
print("Final feature bank shape:", feature_bank.shape)

# Optionally save
torch.save(feature_bank, "feature_bank2.pt")
