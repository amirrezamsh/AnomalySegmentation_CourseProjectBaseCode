############################################### apply pca to features ################################################
import os
import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib

# ========================
# 🔧 Config
# ========================
CHUNKS_DIR = "/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/AnomalySegmentation_CourseProjectBaseCode-main/eval/output"
OUTPUT_FEATURES_PATH = "/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/outputFeatures/reduced_features_80d.npy"
OUTPUT_PCA_MODEL_PATH = "/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/outputFeatures/pca_model_80d.pkl"
TARGET_DIM = 80
NUM_CHUNKS = 6

# ========================
# 🧠 Initialize PCA
# ========================
ipca = IncrementalPCA(n_components=TARGET_DIM)

# ========================
# 🌀 First pass: Fit PCA
# ========================
for i in range(1, NUM_CHUNKS + 1):
    path = os.path.join(CHUNKS_DIR, f"features_chunk{i}.pt")
    data = torch.load(path)

    # --- ADD THESE PRINT STATEMENTS ---
    print(f"\n--- Debugging Chunk {i} ---")
    print(f"Type of 'data' after torch.load(): {type(data)}")

    if isinstance(data, dict):
        print(f"  'data' is a dictionary. Keys: {data.keys()}")
        if "features" in data:
            feats = data["features"]
            print(f"  'features' key found. Type: {type(feats)}, Shape: {feats.shape}")
        else:
            print(f"  WARNING: 'features' key not found in dictionary.")
            # Handle this case, maybe print all values if you're unsure which is features
            for k, v in data.items():
                if torch.is_tensor(v):
                    print(f"    Key '{k}' contains a tensor of shape: {v.shape}")
            continue  # Skip to next chunk if 'features' isn't found
    elif torch.is_tensor(data):
        feats = data  # Directly use 'data' as features
        print(f"  'data' is directly a Tensor. Shape: {feats.shape}")
    else:
        print(f"  ERROR: 'data' is neither a dictionary nor a tensor. Type: {type(data)}")
        continue  # Skip to next chunk as we can't process this type

    # --- END ADDED PRINT STATEMENTS ---

    if feats.ndim > 2:
        feats = feats.view(-1, feats.shape[-1])
        print(f"  Flattened 'feats' shape: {feats.shape}")  # Print after flattening

    ipca.partial_fit(feats.numpy())
    print(f"✅ Fitted PCA on chunk {i}, shape: {feats.shape}")

# ========================
# 🌀 Second pass: Transform
# ========================
reduced_all = []
for i in range(1, NUM_CHUNKS + 1):
    path = os.path.join(CHUNKS_DIR, f"features_chunk{i}.pt")
    data = torch.load(path)

    # Apply the same logic as in the first pass to get 'feats'
    if isinstance(data, dict):
        if "features" in data:
            feats = data["features"]
        else:
            print(f"Skipping transform for chunk {i}: 'features' key not found in dictionary.")
            continue
    elif torch.is_tensor(data):
        feats = data
    else:
        print(f"Skipping transform for chunk {i}: Unexpected data type: {type(data)}")
        continue

    if feats.ndim > 2:
        feats = feats.view(-1, feats.shape[-1])

    reduced = ipca.transform(feats.numpy())
    reduced_all.append(reduced)
    print(f"📦 Transformed chunk {i}, shape: {reduced.shape}")

# ========================
# 💾 Save
# ========================
reduced_all = np.concatenate(reduced_all, axis=0)
np.save(OUTPUT_FEATURES_PATH, reduced_all)
joblib.dump(ipca, OUTPUT_PCA_MODEL_PATH)

print(f"\n✅ Saved reduced features to: {OUTPUT_FEATURES_PATH}")
print(f"✅ Saved PCA model to: {OUTPUT_PCA_MODEL_PATH}")

import numpy as np
data = np.load("/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/outputFeatures/reduced_features_80d.npy")
print(f"Shape of the final reduced features array: {data.shape}")
