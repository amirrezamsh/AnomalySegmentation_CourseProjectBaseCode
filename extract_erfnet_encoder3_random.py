# This script extracts feature representations from three stages (early, mid, and final) of the ERFNet encoder,
# for all training images in the Cityscapes dataset. It processes each image without applying PCA
# and instead performs random sampling of pixel-level features to reduce dimensionality and memory usage.

import os
import random
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from eval.erfnet import ERFNet
from train.main import MyCoTransform

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20
encoder_feature_dim = 128
samples_per_image = 1000  # Number of random pixels to sample
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



        C_e, H_e, W_e = early_feat.shape
        C_m, H_m, W_m = mid_feat.shape
        C_f, H_f, W_f = final_feat.shape

        early_feat = early_feat.view(C_e, -1).permute(1, 0)  # [H*W, C]
        mid_feat = mid_feat.view(C_m, -1).permute(1, 0)  # [H*W, C]
        final_feat = final_feat.view(C_f, -1).permute(1, 0)  # [H*W, C]


        total_pixels = early_feat.shape[0]
        indices = random.sample(range(total_pixels), min(samples_per_image, total_pixels))
        sampled_early_feats = early_feat[indices]  # [N, C]

        total_pixels = mid_feat.shape[0]
        indices = random.sample(range(total_pixels), min(samples_per_image, total_pixels))
        sampled_mid_feats = mid_feat[indices]  # [N, C]

        total_pixels = final_feat.shape[0]
        indices = random.sample(range(total_pixels), min(samples_per_image, total_pixels))
        sampled_final_feats = final_feat[indices]  # [N, C]


        all_early_features.append(sampled_early_feats)
        all_mid_features.append(sampled_mid_feats)
        all_final_features.append(sampled_final_feats)

# --- COMBINE ---
early_feature_bank = torch.cat(all_early_features, dim=0)  # [num_images * N, C]
mid_feature_bank = torch.cat(all_mid_features, dim=0)  # [num_images * N, C]
final_feature_bank = torch.cat(all_final_features, dim=0)  # [num_images * N, C]


print("Final Early feature bank shape:", early_feature_bank.shape)
print("Final Mid feature bank shape:", mid_feature_bank.shape)
print("Final final feature bank shape:", final_feature_bank.shape)

file_prefix = "features_erfnet_encoder3_random"

torch.save({
        'early': early_feature_bank,
        'mid': mid_feature_bank,
        'final': final_feature_bank
    }, f"{file_prefix}.pt")

