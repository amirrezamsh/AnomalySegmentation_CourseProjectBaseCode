
import torch
from sklearn.decomposition import PCA
import numpy as np
from argparse import ArgumentParser
import os 
import sys
import joblib



def compute_covariance(features, mean, reg=1e-5):
    """
    Computes the regularized covariance matrix of feature vectors.

    Args:
        features (Tensor): [N, C] feature matrix
        mean (Tensor): [C] mean vector
        reg (float): regularization strength (small positive value)

    Returns:
        cov (Tensor): [C, C] regularized covariance matrix
    """
    zero_mean = features - mean
    N = features.size(0)
    
    # Standard covariance computation
    cov = (zero_mean.T @ zero_mean) / (N - 1)
    
    # Add regularization (scaled identity matrix)
    cov += reg * torch.eye(cov.size(0), device=features.device, dtype=features.dtype)
    
    return cov

def main():
 
    parser = ArgumentParser()
    parser.add_argument('--features',type=str)
    parser.add_argument('--pcamid',default=None, type=int)
    parser.add_argument('--pcafinal',default=None, type=int)
    args = parser.parse_args()

    if 'random' in args.features :
        sampling_method = 'random'
    elif 'strided' in args.features :
        sampling_method = 'strided'


    
    try :
        data = torch.load(args.features)  # or however you saved it
        print("✅ Successfully loaded the features")
    except :
        print("❌ Error loading features") 
        sys.exit(1)
    
    early_feat = data['early']    # example shape: [2975000, 16]
    mid_feat   = data['mid']      # shape: [2975000, 64]
    final_feat = data['final']    # shape: [2975000, 128]

    #Optionally applting PCA

    pca_mid_value = ''
    if args.pcamid :
        pca_mid = PCA(n_components=args.pcamid, whiten=True)
        mid_feat = pca_mid.fit_transform(mid_feat.numpy())
        print("PCA applied to mid features")
        pca_mid_value = f'_pca{args.pcamid}'


    pca_final_value = ''
    if args.pcafinal :
        pca_final = PCA(n_components=args.pcafinal, whiten=True)
        final_feat = pca_final.fit_transform(final_feat.numpy())
        print("PCA applied to final features")
        pca_final_value = f'_{args.pcafinal}'

    #convert back to tensor

    if isinstance(mid_feat, np.ndarray):
        mid_feat = torch.from_numpy(mid_feat)
        
    if isinstance(final_feat, np.ndarray):
        final_feat = torch.from_numpy(final_feat)


    # Compute mean vector (shape: [128])
    mean_early_vector = torch.mean(early_feat, dim=0)
    mean_mid_vector = torch.mean(mid_feat, dim=0)
    mean_final_vector = torch.mean(final_feat, dim=0)


    # Compute covariance matrix (shape: [C, C])
    cov_early = compute_covariance (early_feat, mean_early_vector)
    cov_mid = compute_covariance (mid_feat, mean_mid_vector)
    cov_final = compute_covariance (final_feat, mean_final_vector)


    print("cov early shape ", cov_early.shape)
    print("cov mid shape ", cov_mid.shape)
    print("cov final shape ", cov_final.shape)

    if args.pcamid :
        mid_pca_path_name = f"midpca_erfnet_encoder3_{sampling_method}_{str(args.midpca)}.pkl"
        joblib.dump(pca_mid, mid_pca_path_name)
        print(f"✅ mid-PCA saved to : {mid_pca_path_name}")

    if args.pcafinal :
        final_pca_path_name = f"finalpca_erfnet_encoder3_{sampling_method}_{str(args.midpca)}.pkl"
        joblib.dump(pca_final, final_pca_path_name)
        print(f"✅ final-PCA saved to : {final_pca_path_name}")


    # # Optionally save for later use
    torch.save({
        'mean_early': mean_early_vector,
        'cov_early': cov_early,
        'mean_mid': mean_mid_vector,
        'cov_mid': cov_mid,
        'mean_final': mean_final_vector,
        'cov_final': cov_final,
    }, f"stats_erfnet_encoder3_{sampling_method}{pca_mid_value}{pca_final_value}.pt")


if __name__ == '__main__' :
    main()