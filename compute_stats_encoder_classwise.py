
############################################### computing mean and inv_covariance ##########################################
import numpy as np
import joblib # For saving the mean and inverse covariance
import os     # For path handling

# ========================
# 🔧 Config
# ========================
# Path to your saved reduced features
REDUCED_FEATURES_PATH = "/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/outputFeatures/reduced_features_80d.npy"

# Paths to save the computed mean and inverse covariance
OUTPUT_MEAN_PATH = "/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/outputFeatures/mean_reduced_80d.npy"
OUTPUT_INV_COV_PATH = "/Users/amirrezarahimi/uni_material/third_semester/advanced_machine_learning/project/outputFeatures/inv_cov_reduced_80d.npy"

# A small regularization term to add to the diagonal of the covariance matrix
# This helps prevent numerical instability and singular matrices, especially
# if some features have very low variance or are perfectly correlated.
REGULARIZATION_EPSILON = 1e-6

# ========================
# 📊 Load Reduced Features
# ========================
print(f"Loading reduced features from: {REDUCED_FEATURES_PATH}")
try:
    reduced_features = np.load(REDUCED_FEATURES_PATH)
    print(f"✅ Successfully loaded reduced features. Shape: {reduced_features.shape}")
except FileNotFoundError:
    print(f"ERROR: Reduced features file not found at {REDUCED_FEATURES_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load reduced features: {e}")
    exit()

# ========================
# 📈 Compute Mean
# ========================
print("Computing mean of reduced features...")
# np.mean(axis=0) computes the mean along the first axis (rows),
# resulting in a mean vector for each feature dimension.
mean_vector = np.mean(reduced_features, axis=0)
print(f"✅ Mean vector computed. Shape: {mean_vector.shape}")

# ========================
# 📉 Compute Covariance Matrix
# ========================
print("Computing covariance matrix of reduced features...")
# np.cov calculates the covariance matrix.
# 'rowvar=False' specifies that each column is a variable (feature)
# and each row is an observation (sample). This is the standard in ML.
covariance_matrix = np.cov(reduced_features, rowvar=False)
print(f"✅ Covariance matrix computed. Shape: {covariance_matrix.shape}")

# ========================
# 🧮 Compute Inverse Covariance Matrix
# ========================
print("Computing inverse covariance matrix...")
# Add a small regularization term to the diagonal to ensure numerical stability
# and prevent issues with singular matrices (e.g., if a feature has zero variance).
regularized_covariance_matrix = covariance_matrix + REGULARIZATION_EPSILON * np.identity(covariance_matrix.shape[0])

try:
    inverse_covariance_matrix = np.linalg.inv(regularized_covariance_matrix)
    print(f"✅ Inverse covariance matrix computed. Shape: {inverse_covariance_matrix.shape}")
except np.linalg.LinAlgError as e:
    print(f"ERROR: Could not compute inverse covariance matrix. It might be singular. Error: {e}")
    print("Consider increasing REGULARIZATION_EPSILON or checking your data for perfectly correlated features.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during inverse covariance calculation: {e}")
    exit()

# ========================
# 💾 Save Mean and Inverse Covariance
# ========================
print("\nSaving computed mean and inverse covariance...")
np.save(OUTPUT_MEAN_PATH, mean_vector)
np.save(OUTPUT_INV_COV_PATH, inverse_covariance_matrix)

print(f"✅ Saved mean vector to: {OUTPUT_MEAN_PATH}")
print(f"✅ Saved inverse covariance matrix to: {OUTPUT_INV_COV_PATH}")

print("\nReady for Mahalanobis Distance calculation!")