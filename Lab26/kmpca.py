#k means algortihm from scratch and using libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


#generate simulated data
#we can use np random function such as normal and uniform
np.random.seed(0)
x = np.random.uniform(10,20,size=(120,50)) #total 120 rows and 50 cols
# print(x.shape)

#now shift the mean create three different classes
#works with just few features also

#class 1; first 40 samples
x[:40,:10] += 3 #scales the first 10 features by 3 for first 40 sample points

#class 2;
x[40:80, 10:30] -= 4 #scales the 10th to 29th feature

#class3 left as it is i.e. centered at zero

# print(x)
#define y as class labels
y = np.repeat([0,1,2],40) #generates 40 copies of each the data listed
print(y)



#b) PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
print(pca.components_)
print(pca.explained_variance_)

###not required
# #draw vectors
# #define a function for drawing vectors
# def draw_vectors(v0, v1, ax=None):
#     ax= ax or plt.gca() #if axes is none, then current axes
#     arrowprops = dict(arrowstyle = '->',
#                       linewidth = 2,
#                       shrinkA = 0,
#                       shrinkB = 0)
#     ax.annotate('',v1,v0,arrowprops=arrowprops)
#
# #plot the arrows
# x_pca = pca.transform(x)
# plt.scatter(x_pca[:,0],x_pca[:,1],alpha=0.2, c=y, cmap='viridis',edgecolor='k')
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector[:2]*3*np.sqrt(length) ##first two columns only not the entire 50 dimensions
#     draw_vectors(np.mean(x_pca, axis=0), np.mean(x_pca, axis=0) + v)
#
# plt.show()

#part b)
# ----- Perform PCA -----
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print("PCA Components:\n", pca.components_)
print("Explained Variance:\n", pca.explained_variance_)

# ----- Plot PC1 vs PC2 with color-coded classes -----
plt.figure(figsize=(8,6))
scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: PC1 vs PC2 with Class Labels")
plt.grid(True)
plt.show()

#
# #c) kmeans
#kmeans using sklearn
#define the value of k
# k = 2
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans3.fit(x)

cluster_labels = kmeans3.labels_
print(f"Cluster labels:\n", kmeans3.labels_)

 #plot the clusters
plt.figure(figsize=(10,7))
plt.scatter(x[:,10], x[:,11], c=kmeans3.labels_, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 10")
plt.ylabel("Feature 11")
plt.title("K-Means Clustering with K=3")
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

#Compare if the generated cluster classes using kmeans is same as the class labels assigned
#pd.crosstab()

print(pd.crosstab(y, cluster_labels , rownames=['Actual'], colnames=['Predicted']))