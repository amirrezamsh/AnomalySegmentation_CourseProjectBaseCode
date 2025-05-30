import torch
from sklearn.decomposition import PCA
import numpy as np
import joblib
from argparse import ArgumentParser
import sys


def main() :
  parser = ArgumentParser()
  parser.add_argument('--features',type=str)
  parser.add_argument('--pca',default=None, type=int)
  args = parser.parse_args()

  if 'random' in args.features :
      sampling_method = 'random'
  elif 'strided' in args.features :
      sampling_method = 'strided'
      

  # feature_bank: shape [N, 128], float tensor
  try :
    feature_bank = torch.load(args.features)  # or however you saved it
    print("✅ Successfully loaded the features")
  except :
     print("❌ Error loading features") 
     sys.exit(1)

  pca_value = ''
  if args.pca :
    pca = PCA(n_components= args.pca)
    feature_bank = pca.fit_transform(feature_bank.numpy())
    print(f"✅ PCA applied to reduce dimensions to {args.pca}")
    pca_value = f"_pca{args.pca}"

  if isinstance(feature_bank, np.ndarray):
          feature_bank = torch.from_numpy(feature_bank)

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

  if args.pca :
    pca_path_name = f"pca_erfnet_encoder_{sampling_method}_{str(args.pca)}.pkl"
    joblib.dump(pca, pca_path_name)
    print(f"✅ PCA saved to : {pca_path_name}")

  # Optionally save for later use
  torch.save({
    "mean" : mean_vector,
    "cov" : covariance_matrix
  },f"stats_erfnet_encoder_{sampling_method}{pca_value}.pt")

if __name__ == '__main__':
    main()
