# Work on NCI data - build classification model after reducing the gene expression features using hierarchical clustering.
# Compare this with the PCA approach
#working on NCI 60 dataset
import pandas as pd
import seaborn as sns
from ISLP import load_data
from matplotlib import pyplot as plt
from matplotlib.lines import lineStyles
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch

#loading the dataset
NCI60 = load_data("NCI60")
NCI60.keys() #returns the keys, here it is data and labs
# print(NCI60['data'].shape) #64,6830
# print(NCI60['labels'].value_counts()) #returns the count of all the labels associated with the cancer cell lines

#extract  the data from the loaded NCI60

data = pd.DataFrame(NCI60['data']) #64, 6830
# print(data)
labels = NCI60['labels'] #64,1
# print(labels)

#preprocessing on the dataset

# #check for missing values in the dataset
# missing_values = data.isnull().sum()
# print(missing_values)
#
# #heatmap to check for correlation between the features
# correlation_matrix = data.corr()
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
# plt.show()

#standardize the data
scaler  = StandardScaler()
data_scaled = scaler.fit_transform(data)

#perform hierarchical clustering
#two methods to keep in mind,
#linkage method- it tells us how to compute the distance between the clusters
#method parameter of linkage allows to select a method of linkage such as complete,single, average etc
#distance metric such as euclidean allows to calculate the distance between the pair of data points

#two methods using scipy and scikit
hc_scipy = sch.linkage(data_scaled, method='complete', metric='euclidean')
print(hc_scipy) #returns the linkage matrix, each row tells how the two clusters where merged at what distance and how many points now it holds in the new cluster
#  [124.         125.         163.48974042  64.        ] last two clusters merged to form a single cluster containing all the 64 datapoints

#to choose the cut size as per the max number of clusters
num_clusters = 4
clusters = sch.fcluster(hc_scipy, t=num_clusters, criterion='maxclust')
print(clusters)
#plot the dendrogram to visualise the clustering efficiently

plt.figure(figsize=(10,7))
dend_hc = dendrogram(hc_scipy,leaf_rotation=90, link_color_func=lambda x:"black") #link color function overrides the default color mapping
plt.title('Hierarchical clustering using scipy')
plt.xlabel('Data points')
plt.ylabel('cluster distances')

#plot a horizontal line to indicate the clustering point
#how to choose
# Look for a large vertical gap between successive merges.
# Pick a horizontal line (y-value) that cuts through that gap
#using the cut height to draw the horizontal line, pass the cut height as y
# The row hc_scipy[-(num_clusters-1), :] gives the merge that reduces the number of clusters from num_clusters to num_clusters-1.

cut_height = hc_scipy[-num_clusters+1,2] - 0.0000001
print(cut_height)
plt.axhline(y=cut_height,color='red',linestyle='--')
plt.show()




































# """
# NCI60 Gene Expression Analysis
# Hierarchical Clustering of Cancer Cell Lines
# """
#
# import pandas as pd
# import matplotlib.pyplot as plt
# from ISLP import load_data
# from scipy.cluster.hierarchy import dendrogram, fcluster
# from sklearn.preprocessing import StandardScaler
# import scipy.cluster.hierarchy as sch
#
#
# def load_and_prepare_data():
#     """Load and prepare the NCI60 dataset"""
#     nci60 = load_data("NCI60")
#     print(f"Data shape: {nci60['data'].shape}")
#     print("Label distribution:\n", nci60['labels'].value_counts())
#
#     data = pd.DataFrame(nci60['data'])
#     labels = nci60['labels']
#     return data, labels
#
#
# def preprocess_data(data):
#     """Standardize the gene expression data"""
#     scaler = StandardScaler()
#     return scaler.fit_transform(data)
#
#
# def perform_clustering(data_scaled, n_clusters=4, method='complete'):
#     """Perform hierarchical clustering and return results"""
#     linkage_matrix = sch.linkage(data_scaled, method=method, metric='euclidean')
#     clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
#     return linkage_matrix, clusters
#
#
# def plot_dendrogram(linkage_matrix, n_clusters, labels=None):
#     """Visualize hierarchical clustering results"""
#     plt.figure(figsize=(12, 7))
#
#     # Plot dendrogram
#     dendrogram(linkage_matrix,
#                leaf_rotation=90,
#                labels=labels.values if labels is not None else None,
#                link_color_func=lambda x: "black")
#
#     # Calculate and plot cut line
#     cut_height = linkage_matrix[-n_clusters + 1, 2] - 1e-5
#     plt.axhline(y=cut_height, color='red', linestyle='--',
#                 label=f'{n_clusters} clusters cutoff')
#
#     plt.title('Hierarchical Clustering of NCI60 Cancer Cell Lines')
#     plt.xlabel('Cell Line Samples' if labels is None else 'Cancer Types')
#     plt.ylabel('Cluster Distance')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     return cut_height
#
#
# def main():
#     # Load and prepare data
#     data, labels = load_and_prepare_data()
#
#     # Preprocess data
#     data_scaled = preprocess_data(data)
#
#     # Perform clustering
#     n_clusters = 4
#     linkage_matrix, clusters = perform_clustering(data_scaled, n_clusters)
#
#     print("\nCluster assignments:")
#     print(pd.Series(clusters).value_counts().sort_index())
#
#     # Visualize results
#     cut_height = plot_dendrogram(linkage_matrix, n_clusters, labels)
#     print(f"\nCut height for {n_clusters} clusters: {cut_height:.4f}")
#
#
# if __name__ == "__main__":
#     main()