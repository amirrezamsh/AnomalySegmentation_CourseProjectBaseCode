################################################ new extracting method using custom class ################################################
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from PIL import Image
from erfnet import ERFNet
from dataset import cityscapes  # your custom Cityscapes class

# ─── Configuration ─────────────────────────────────────────────────────────
DATA_ROOT    = '/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/cityscapes/'
WEIGHTS_PATH = '/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/AnomalySegmentation_CourseProjectBaseCode-main/trained_models/erfnet_pretrained.pth'
OUTPUT_DIR   = '/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/AnomalySegmentation_CourseProjectBaseCode-main/eval/output'
BATCH_SIZE   = 2
IGNORE_LABEL = 19

CHUNK_START  = 2500
CHUNK_END    = 2975   # exactly last image index + 1
NUM_IMAGES   = CHUNK_END - CHUNK_START  # 475 images

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Transforms ────────────────────────────────────────────────────────────
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])

def transform_label(label):
    arr = np.array(label, dtype=np.int64)
    arr[arr == 255] = IGNORE_LABEL
    return torch.from_numpy(arr)

# ─── Dataset & Loader ──────────────────────────────────────────────────────
dataset = cityscapes(DATA_ROOT, input_transform=input_transform_cityscapes,
                     target_transform=transform_label, subset='train')

loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=0)

# ─── Load Model ────────────────────────────────────────────────────────────
model = ERFNet(num_classes=20)
state = torch.load(WEIGHTS_PATH, map_location='cpu')
state_dict = state.get('state_dict', state)
model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
model.eval()

# ─── Extract & Save Last Chunk ──────────────────────────────────────────────
feats_list, labels_list = [], []
processed = 0

loader_iter = iter(loader)

# Skip full batches before CHUNK_START
num_full_batches_to_skip = CHUNK_START // BATCH_SIZE
for _ in range(num_full_batches_to_skip):
    next(loader_iter)

# Partial skip inside next batch if needed
partial_skip = CHUNK_START % BATCH_SIZE
if partial_skip > 0:
    images, labels, _, _ = next(loader_iter)
    images = images[partial_skip:]
    labels = labels[partial_skip:]
    if images.size(0) > 0:
        with torch.no_grad():
            feats = model.encoder(images)
            B, C, Hf, Wf = feats.shape
            lbl_ds = F.interpolate(labels.unsqueeze(1).float(),
                                   size=(Hf, Wf), mode='nearest').long().squeeze(1)
            feats_flat  = feats.permute(0,2,3,1).reshape(-1, C)
            labels_flat = lbl_ds.reshape(-1)
            valid_mask = (labels_flat != IGNORE_LABEL)
            feats_filtered  = feats_flat[valid_mask]
            labels_filtered = labels_flat[valid_mask]

            feats_list.append(feats_filtered.cpu())
            labels_list.append(labels_filtered.cpu())

            processed += images.size(0)
            print(f"🔁 Processed partial batch after skip: {processed}/{NUM_IMAGES} images | "
                  f"Batch output shape: {feats_filtered.shape}")

# Process remaining full batches
with torch.no_grad():
    for images, labels, _, _ in tqdm(loader_iter, desc=f"Chunk 6 ({CHUNK_START}–{CHUNK_END - 1})"):
        if processed >= NUM_IMAGES:
            break

        if processed + images.size(0) > NUM_IMAGES:
            images = images[:NUM_IMAGES - processed]
            labels = labels[:NUM_IMAGES - processed]

        feats = model.encoder(images)
        B, C, Hf, Wf = feats.shape

        lbl_ds = F.interpolate(labels.unsqueeze(1).float(),
                               size=(Hf, Wf), mode='nearest').long().squeeze(1)

        feats_flat  = feats.permute(0,2,3,1).reshape(-1, C)
        labels_flat = lbl_ds.reshape(-1)

        valid_mask = (labels_flat != IGNORE_LABEL)
        feats_filtered  = feats_flat[valid_mask]
        labels_filtered = labels_flat[valid_mask]

        feats_list.append(feats_filtered.cpu())
        labels_list.append(labels_filtered.cpu())

        processed += images.size(0)

        print(f"🔁 Processed {processed}/{NUM_IMAGES} images | Batch output shape: {feats_filtered.shape}")

chunk_feats  = torch.cat(feats_list, dim=0)
chunk_labels = torch.cat(labels_list, dim=0)

torch.save(chunk_feats,  os.path.join(OUTPUT_DIR, 'features_chunk6.pt'))
torch.save(chunk_labels, os.path.join(OUTPUT_DIR, 'labels_chunk6.pt'))

print(f"\n✅ Saved last chunk: {chunk_feats.shape[0]} valid pixels → features_chunk6.pt / labels_chunk6.pt")

