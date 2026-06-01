# hierarchical_clustering_usarrests.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Load USArrests dataset from online R mirror
def load_usarrests(data):
    df = pd.read_csv(data, index_col=0)  # Add index_col=0 here!
    return df


# Normalize the data using z-score standardization
def normalize_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)

def hierarchical_clustering(data, method='complete', metric='euclidean', title=''):
    # Compute distance matrix
    dist = pdist(data, metric=metric)  # <- works for euclidean, correlation, etc.

    # Linkage method
    link = linkage(dist, method=method)

    # Plot dendrogram
    plt.figure(figsize=(12, 5))
    dendrogram(link, labels=data.index.tolist(), leaf_rotation=90)
    plt.title(f"{title} Dendrogram ({metric.capitalize()} Distance)")
    plt.xlabel("States")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    # Cut dendrogram into clusters
    clusters = fcluster(link, t=3, criterion='maxclust')

    # Group state names into clusters
    cluster_dict = {}
    for i, state in enumerate(data.index):
        cluster_dict.setdefault(clusters[i], []).append(state)

    return cluster_dict

# Print readable output of clusters
def print_clusters(cluster_dict, label):
    print(f"\nClusters using {label}:\n")
    for cluster_id, states in cluster_dict.items():
        print(f"Cluster {cluster_id} ({len(states)} states):")
        print(", ".join(states))
        print()

# Main function to execute both clustering types
def main():
    # Load and normalize data
    data = "~/Downloads/USArrests.csv"
    df = load_usarrests(data)
    df_norm = normalize_data(df)

    # (a) Euclidean Distance + Complete Linkage
    clusters_euclidean = hierarchical_clustering(df_norm, method='complete', metric='euclidean',
                                                  title="Complete Linkage + Euclidean")
    print_clusters(clusters_euclidean, "Euclidean Distance")

    # (c) Correlation Distance + Complete Linkage
    clusters_correlation = hierarchical_clustering(df_norm, method='complete', metric='correlation',
                                                    title="Complete Linkage + Correlation")
    print_clusters(clusters_correlation, "Correlation Distance")

if __name__ == "__main__":
    main()
