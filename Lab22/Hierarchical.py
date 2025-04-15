import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from ISLP import load_data
from ISLP.cluster import compute_linkage

# Load and scale data
NCI60 = load_data('NCI60')
nci_data = NCI60['data'] #(64, 6830)
nci_labs = NCI60['labels'] #(64,1)

#Standardization - zero mean and unit variance.
scaler = StandardScaler()
nci_scaled = scaler.fit_transform(nci_data)

# Dendrogram plotting function
def plot_nci(linkage, ax, cut=-np.inf):
    hc = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage=linkage.lower()
    ).fit(nci_scaled)

    linkage_matrix = compute_linkage(hc)
    #compute_linkage(hc) converts the clustering object to a linkage matrix compatible with scipy.dendrogram().

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=np.asarray(nci_labs),
        leaf_font_size=10,
        color_threshold=cut,
        above_threshold_color='black'
    )

    ax.set_title(f'{linkage} Linkage')
    return hc

# Plot dendrograms for three linkage types
fig, axes = plt.subplots(3, 1, figsize=(15, 25))

plot_nci('Complete', axes[0])
plot_nci('Average', axes[1])
plot_nci('Single', axes[2])

plt.tight_layout()
plt.show()
