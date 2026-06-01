import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples_per_class=40, n_features=50, seed=42):
    """
    Generate simulated data for 3 classes.
    Each class has a different mean to make them distinguishable.
    Returns the data matrix X and the true class labels y.
    """
    np.random.seed(seed)

    # Class 1: mean = 0
    class1 = np.random.normal(loc=0, scale=1, size=(n_samples_per_class, n_features))

    # Class 2: mean = 3
    class2 = np.random.normal(loc=3, scale=1, size=(n_samples_per_class, n_features))

    # Class 3: mean = -3
    class3 = np.random.normal(loc=-3, scale=1, size=(n_samples_per_class, n_features))

    # Stack all data vertically
    X = np.vstack([class1, class2, class3])

    # Create true class labels: 0, 1, 2
    y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)

    return X, y


def standardize_data(X):
    """
    Standardize the dataset using zero mean and unit variance.
    This step is important before applying PCA and K-means.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def perform_pca(X_scaled, y_true):
    """
    Perform PCA on the scaled data and plot the first two principal components.
    Points are colored by their true class labels.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting the PCA result
    plt.figure(figsize=(8, 6))
    for label in np.unique(y_true):
        plt.scatter(X_pca[y_true == label, 0], X_pca[y_true == label, 1], label=f'Class {label}')
    plt.title("PCA: First 2 Principal Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

    return X_pca


def perform_kmeans(X_scaled, y_true, n_clusters=3):
    """
    Run K-means clustering and compare the results to the true labels.
    Displays a confusion-matrix-style table using pandas crosstab.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Compare K-means cluster labels to true class labels
    confusion = pd.crosstab(y_true, kmeans_labels, rownames=['True'], colnames=['K-means'])
    print("\nK-means vs True Labels (Confusion Matrix):")
    print(confusion)

    return kmeans_labels


def main():
    """
    Main function to run all steps:
    - Generate data
    - Standardize data
    - Perform PCA and visualize
    - Run K-means and evaluate clustering
    """
    # Step 1: Generate simulated dataset
    X, y_true = generate_data()

    # Step 2: Standardize the dataset
    X_scaled = standardize_data(X)

    # Step 3: PCA visualization
    perform_pca(X_scaled, y_true)

    # Step 4: K-means clustering and result comparison
    perform_kmeans(X_scaled, y_true)


# Run the main function
if __name__ == "__main__":
    main()
