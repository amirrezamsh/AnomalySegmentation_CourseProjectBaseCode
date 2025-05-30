import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np

data = torch.load("./feature_bank2.pt")  # or however you saved it

# early_feat = data['early']    # shape: [2975000, 16]
# mid_feat   = data['mid']      # shape: [2975000, 64]
# final_feat = data['final']    # shape: [2975000, 128]

# Assume `mid_feat_np` is shape [1000, 64]
pca = PCA()
pca.fit(data)

explained = np.cumsum(pca.explained_variance_ratio_) * 100

plt.plot(explained)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance (%)')
plt.grid(True)
plt.axhline(y=95, color='r', linestyle='--')
plt.title('Explained variance for mid features')
plt.show()
