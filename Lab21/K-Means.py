import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#  Generate Synthetic Data

np.random.seed(0)
X = np.random.randn(50, 2)
X[:25, 0] += 3   # Shift first 25 points in x-direction
X[:25, 1] -= 4   # Shift first 25 points in y-direction


#  K-Means using scikit-learn
# K=2 clustering
kmeans2 = KMeans(n_clusters=2, random_state=2, n_init=20).fit(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans2.labels_, cmap='coolwarm', edgecolors='k')
plt.title("K-Means Clustering with K=2")
plt.show()

# K=3 clustering
kmeans3 = KMeans(n_clusters=3, random_state=3, n_init=20).fit(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans3.labels_, cmap='coolwarm', edgecolors='k')
plt.title("K-Means Clustering with K=3")
plt.show()

# Comparing inertia (distortion) for n_init=1 vs n_init=20
kmeans3_n1 = KMeans(n_clusters=3, random_state=3, n_init=1).fit(X)
kmeans3_n20 = KMeans(n_clusters=3, random_state=3, n_init=20).fit(X)
print("Inertia with n_init=1:", kmeans3_n1.inertia_)
print("Inertia with n_init=20:", kmeans3_n20.inertia_)


#  Manual K-Means Implementation (No class)


def initialize_centroids(X, k, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def compute_distances(X, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def assign_clusters(X, centroids):
    distances = compute_distances(X, centroids)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([
        X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else np.zeros(X.shape[1])
        for i in range(k)
    ])

def has_converged(old_centroids, new_centroids, tol=1e-4):
    return np.linalg.norm(old_centroids - new_centroids) < tol

# Set parameters
k = 2
max_iter = 300
random_state = 42

# Run K-Means manually
centroids = initialize_centroids(X, k, random_state)

for i in range(max_iter):
    labels = assign_clusters(X, centroids)
    new_centroids = update_centroids(X, labels, k)
    if has_converged(centroids, new_centroids):
        break
    centroids = new_centroids

# Plot manual implementation result
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', label='Centroids')
plt.title("Manual K-Means Clustering (K=2)")
plt.legend()
plt.show()
