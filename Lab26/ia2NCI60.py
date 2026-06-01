import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load NC160 data
NC160 = load_data('NC160')
X = NC160.iloc[:, 1:].values  # Assuming first column is cancer type
y = NC160.iloc[:, 0].values   # Cancer types

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca_scores = pca.fit_transform(X_scaled)

# 2a. Plot first three principal components
plt.figure(figsize=(15, 5))

# PC1 vs PC2
plt.subplot(131)
for cancer_type in np.unique(y):
    idx = y == cancer_type
    plt.scatter(pca_scores[idx, 0], pca_scores[idx, 1], label=cancer_type)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# PC1 vs PC3
plt.subplot(132)
for cancer_type in np.unique(y):
    idx = y == cancer_type
    plt.scatter(pca_scores[idx, 0], pca_scores[idx, 2], label=cancer_type)
plt.xlabel('PC1')
plt.ylabel('PC3')

# PC2 vs PC3
plt.subplot(133)
for cancer_type in np.unique(y):
    idx = y == cancer_type
    plt.scatter(pca_scores[idx, 1], pca_scores[idx, 2], label=cancer_type)
plt.xlabel('PC2')
plt.ylabel('PC3')

plt.tight_layout()
plt.show()

# 2b. Plot variance explained
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.subplot(122)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained')
plt.tight_layout()
plt.show()

# 2c. Hierarchical clustering on first few PCs
n_components = 5  # Use first 5 PCs
Z = linkage(pca_scores[:, :n_components], method='complete')

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=y, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Distance')
plt.show()