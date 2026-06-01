#NCI60 data
#clustering on the features
import numpy as np
import pandas as pd
from ISLP import load_data
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import distance_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch



NCI60 = load_data('NCI60')
# print(NCI60.keys())
data = NCI60['data']
labels = NCI60['labels'].to_numpy().ravel()
# print(data.shape)

#preprocessing
#check for the null values
#check for correlation between the data points

#transpose the data as the clustering by defualt will occur on the samples and we want the gene expression features to
# be clustered

features = np.transpose(data)
# print(features.shape) #6840,64

#standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
print(scaled_data)

#perform hierarchical clustering on features

hc_features = sch.linkage(scaled_data,
                          method='complete',
                          metric='euclidean')

# print(hc_features.shape)

max_clusters = 4
clusters = fcluster(hc_features,t=max_clusters,criterion='maxclust')
print(clusters.shape)

#get the cut height in the form of the  number of clusters needed
cut_height = hc_features[-max_clusters+1, 2] - 10**-6


#plot the dendrogram
# hc_dendrogram  = sch.dendrogram(hc_features,link_color_func=lambda x:"black")
# plt.title("Heirarchical clustering on the gene expression levels")
# plt.xlabels("Gene expressions")
# plt.ylabels("Cluster distance")
# plt.axhline(y=cut_height, color='red',linestyle='--')
# # plt.show()

#reduce the gene expression features by replacing features with average of clusters, as each feature belongs to certain cluster
original_data = data

#create a dataframe for cluster assignment i.e. which cluster each gene belongs to

cluster_df = pd.DataFrame({'gene_index':range(len(clusters)),'cluster':clusters})
print(cluster_df)

#group the genes by cluster
reduced_data = []
for i in range(1, max_clusters+1):
    gene_indices = cluster_df[cluster_df['cluster']==i]['gene_index'].values ##array of integers representing the positions (columns) in the original data corresponding to genes in cluster i.
    cluster_avg = np.mean(original_data[:,gene_indices], axis=1) #cluster_avg of shape (64,), one average value per sample of the gene expression features corresponding to one cluster
    reduced_data.append(cluster_avg)


reduced_data = np.array(reduced_data).T

print(reduced_data.shape) #64,4 where each column represents the cluster average of the genes present in it i.e. one feature per cluster


#train a model on this reduced data
#use a classifier
#split the dataset

x_train, x_test, y_train, y_test = train_test_split(reduced_data,labels,test_size = 0.3, random_state=42 )

clf = LogisticRegression(max_iter=1000)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(f"Accuracy classification:", accuracy_score(y_test,y_pred))

























#
# # NCI60 data - Clustering of gene expression features
# import numpy as np
# from ISLP import load_data
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import fcluster, dendrogram
# from sklearn.preprocessing import StandardScaler
# import scipy.cluster.hierarchy as sch
#
# # Load data
# NCI60 = load_data('NCI60')
# data = NCI60['data']
# labels = NCI60['labels']
#
# # Cluster features (genes) rather than samples
# features = data.T  # Transpose to get genes as rows (6830 genes × 64 samples)
#
# # Standardize the data (gene-wise standardization)
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(features)
#
# # Perform hierarchical clustering on features
# hc_features = sch.linkage(scaled_data,
#                          method='complete',
#                          metric='euclidean')
#
# # Cluster assignments
# max_clusters = 4
# clusters = fcluster(hc_features, t=max_clusters, criterion='maxclust')
# print(f"Cluster assignments:\n{np.unique(clusters, return_counts=True)}")
#
# # Calculate cut height
# cut_height = hc_features[-max_clusters + 1, 2] - 1e-5
#
# # Plot dendrogram
# plt.figure(figsize=(12, 6))
# dendrogram(hc_features,
#           link_color_func=lambda x: "black",
#           orientation='top',
#           labels=None)  # Too many genes to show individual labels
#
# plt.title("Hierarchical Clustering of Gene Expression Features (NCI60)")
# plt.xlabel("Genes (6830 total)")
# plt.ylabel("Cluster Distance")
# plt.axhline(y=cut_height, color='red', linestyle='--',
#            label=f'Cut for {max_clusters} clusters')
# plt.legend()
# plt.tight_layout()
# plt.show()