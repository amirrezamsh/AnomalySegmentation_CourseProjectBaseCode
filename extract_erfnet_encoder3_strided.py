#this file does not do random sampling but something like a uniform sampling with default stride of 4 

import os
import random
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from eval.erfnet import ERFNet
from train.main import MyCoTransform


def uniform_sample(feat_map, stride = 4):
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

all_early_features = []
all_mid_features = []
all_final_features = []

# --- MAIN LOOP ---
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Extracting features"):
        pil_img = Image.open(img_path).convert("RGB")
        dummy_gt = Image.new("L", pil_img.size)  # dummy GT

        input_tensor, _ = co_transform(pil_img, dummy_gt)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        early_feat, mid_feat, final_feat = model(input_tensor, multi_encode = True)  # shape: [C, H, W].squeeze(0).cpu()

        early_feat = early_feat.squeeze(0).cpu()
        mid_feat = mid_feat.squeeze(0).cpu()
        final_feat = final_feat.squeeze(0).cpu()



        early_feat = uniform_sample(early_feat)
        mid_feat = uniform_sample(mid_feat)
        final_feat = uniform_sample(final_feat)


        all_early_features.append(early_feat)
        all_mid_features.append(mid_feat)
        all_final_features.append(final_feat)

# --- COMBINE ---
early_feature_bank = torch.cat(all_early_features, dim=0)  # [num_images * N, C]
mid_feature_bank = torch.cat(all_mid_features, dim=0)  # [num_images * N, C]
final_feature_bank = torch.cat(all_final_features, dim=0)  # [num_images * N, C]


print("Final Early feature bank shape:", early_feature_bank.shape)
print("Final Mid feature bank shape:", mid_feature_bank.shape)
print("Final final feature bank shape:", final_feature_bank.shape)

file_prefix = "features_erfnet_encoder3_strided"

torch.save({
        'early': early_feature_bank,
        'mid': mid_feature_bank,
        'final': final_feature_bank
    }, f"{file_prefix}.pt")

