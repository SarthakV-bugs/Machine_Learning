import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([
    [1, 4],
    [1, 3],
    [0, 4],
    [5, 1],
    [6, 2],
    [4, 0]
])

# 3a. Plot observations
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='black')
for i in range(len(X)):
    plt.text(X[i, 0]+0.1, X[i, 1]+0.1, f'Obs {i+1}')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Initial Observations')
plt.grid()
plt.show()

# 3b. Random cluster assignment
np.random.seed(42)  # For reproducibility
labels = np.random.choice([0, 1], size=len(X))
print("Initial cluster labels:", labels)

# Function to plot with colors
def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue']
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]])
        plt.text(X[i, 0]+0.1, X[i, 1]+0.1, f'Obs {i+1}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.grid()
    plt.show()

plot_clusters(X, labels, "Initial Random Clustering")

# Function to compute centroids
def compute_centroids(X, labels):
    centroids = []
    for k in np.unique(labels):
        centroids.append(np.mean(X[labels == k], axis=0))
    return np.array(centroids)

# Function to assign labels
def assign_labels(X, centroids):
    distances = np.zeros((len(X), len(centroids)))
    for k in range(len(centroids)):
        distances[:, k] = np.sqrt(np.sum((X - centroids[k])**2, axis=1))
    return np.argmin(distances, axis=1)

# 3c. Compute centroids
centroids = compute_centroids(X, labels)
print("Initial centroids:", centroids)

# 3d. Assign new labels
new_labels = assign_labels(X, centroids)
print("Updated cluster labels:", new_labels)
plot_clusters(X, new_labels, "After First Update")

# 3e. Repeat until convergence
old_labels = None
while not np.array_equal(new_labels, old_labels):
    old_labels = new_labels.copy()
    centroids = compute_centroids(X, old_labels)
    new_labels = assign_labels(X, centroids)
    print("Updated centroids:", centroids)
    print("Updated labels:", new_labels)
    plot_clusters(X, new_labels, "Iterative Update")

# 3f. Final plot
plot_clusters(X, new_labels, "Final Clustering")
print("Final cluster assignments:")
for i in range(len(X)):
    print(f"Observation {i+1}: Cluster {new_labels[i]+1}")