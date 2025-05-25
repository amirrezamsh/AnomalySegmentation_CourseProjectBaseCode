import torch

# feature_bank: shape [N, 128], float tensor
feature_bank = torch.load("feature_bank.pt")  # or however you saved it

# Compute mean vector (shape: [128])
mean_vector = torch.mean(feature_bank, dim=0)

# Center features by subtracting mean
centered_features = feature_bank - mean_vector.unsqueeze(0)  # [N, 128]

# Compute covariance matrix (shape: [128, 128])
# Using the unbiased estimator (N-1 in denominator)
covariance_matrix = (centered_features.t() @ centered_features) / (feature_bank.shape[0] - 1)

# Apply small regularization to covariance matrix to ensure it is invertible
reg_lambda = 1e-6
identity = torch.eye(covariance_matrix.shape[0])
covariance_matrix += reg_lambda * identity

print("Mean vector shape:", mean_vector.shape)
print("Covariance matrix shape:", covariance_matrix.shape)

# Optionally save for later use
torch.save(mean_vector, "mean_vector.pt")
torch.save(covariance_matrix, "covariance_matrix.pt")
