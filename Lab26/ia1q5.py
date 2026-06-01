import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import pairwise_distances
import numpy as np

# Load dataset
data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USArrests.csv', index_col=0)

# (a) Euclidean + Complete Linkage
Z_euc = linkage(data, method='complete', metric='euclidean')
dendrogram(Z_euc, labels=data.index, leaf_rotation=90)
plt.title("Dendrogram (Euclidean)")
plt.show()

# (b) Cut into 3 clusters
labels_euc = fcluster(Z_euc, t=3, criterion='maxclust')
print("Cluster labels (Euclidean):")
print(pd.Series(labels_euc, index=data.index))

# (c) Correlation + Complete Linkage
corr_dist = 1 - np.corrcoef(data.T).T @ np.corrcoef(data.T)
dist_matrix = pairwise_distances(data, metric='correlation')
Z_corr = linkage(dist_matrix, method='complete')
dendrogram(Z_corr, labels=data.index, leaf_rotation=90)
plt.title("Dendrogram (Correlation)")
plt.show()

# (d) Cut into 3 clusters
labels_corr = fcluster(Z_corr, t=3, criterion='maxclust')
print("Cluster labels (Correlation):")
print(pd.Series(labels_corr, index=data.index))

# (e) Comment:
# Euclidean and correlation-based distances often yield similar clusters when data is standardized,
# because they both reflect relationships between variables. Euclidean is sensitive to scale, but when scaled,
# both measures give comparable groupings.
