#1
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
def read_data(n_samples=100, noise=0.3, random_state=42):
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
def train_svm_models(X_train, y_train):
    poly_svc = SVC(kernel='poly', degree=2, C=1.0, random_state=42)
    poly_svc.fit(X_train, y_train)
    rbf_svc = SVC(kernel='rbf', gamma=1.0, C=1.0, random_state=42)
    rbf_svc.fit(X_train, y_train)
    return poly_svc, rbf_svc
def evaluate_models(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        print(f"\n{name} Kernel")
        print(f"Train Accuracy: {train_acc:.2f} | Test Accuracy: {test_acc:.2f}")
        print(f"Train F1 Score: {train_f1:.2f} | Test F1 Score: {test_f1:.2f}")
def plot_data(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=50, cmap='coolwarm')
    plt.title("Training Data")
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', s=50, cmap='coolwarm')
    plt.title("Test Data")
    plt.tight_layout()
    plt.show()
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=30, cmap='coolwarm')
    plt.title(title)
    plt.tight_layout()
    plt.show()
def main():
    X, y = read_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    poly_svc, rbf_svc = train_svm_models(X_train, y_train)
    evaluate_models({'Polynomial': poly_svc, 'RBF': rbf_svc},
                    X_train, y_train, X_test, y_test)
    plot_data(X_train, y_train, X_test, y_test)
    plot_decision_boundary(poly_svc, X_train, y_train, "Polynomial Kernel Decision Boundary")
    plot_decision_boundary(rbf_svc, X_train, y_train, "RBF Kernel Decision Boundary")
if __name__ == "__main__":
    main()



#2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.linear_model import LogisticRegression
from scipy.cluster.hierarchy import linkage, dendrogram
def read_and_describe_data():
    from ISLP import load_data
    NCI60 = load_data('NCI60')
    X = NCI60['data']
    labels = NCI60['labels']
    # Convert labels to 1D (if they are not already)
    if isinstance(labels, pd.DataFrame):
        labels = labels.squeeze()  # from DataFrame to Series
    elif isinstance(labels, np.ndarray) and labels.ndim > 1:
        labels = labels.ravel()  # flatten to 1D array

    print("\n🔍 Dataset Information")
    print("-" * 40)
    print("Shape of Data:", X.shape)
    print("\nData Description:\n", pd.DataFrame(X).describe())
    print("\nLabel Distribution:\n", pd.Series(labels).value_counts())

    return X, labels
def perform_eda(X, labels):
    df = pd.DataFrame(X)
    df['CancerType'] = labels
    # Plotting label distribution
    plt.figure(figsize=(10, 4))
    sns.countplot(y='CancerType', hue='CancerType', data=df, order=df['CancerType'].value_counts().index,
                  palette='tab20', legend=False)
    plt.title('Distribution of Cancer Types (Target Variable)')
    plt.xlabel('Count')
    plt.ylabel('Cancer Type')
    plt.tight_layout()
    plt.show()
    print(" Interpretation: This shows the dataset is imbalanced across cancer types.")

    # Correlation heatmap of first 10 features for quick EDA
    plt.figure(figsize=(10, 8))
    corr = df.iloc[:, :10].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of First 10 Features')
    plt.show()
    print(" Interpretation: Highlights potential redundant or highly correlated features.")
def apply_pca_and_clustering(X, labels):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Create PCA dataframe
    pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pca_df['CancerType'] = labels

    # PCA Scatter Plots
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='CancerType', palette='tab20', s=70)
    plt.title('PC1 vs PC2 - Colored by Cancer Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    print("Interpretation: PCA plot shows how well the first two components separate different cancer types.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC3', hue='CancerType', palette='tab20', s=70)
    plt.title('PC1 vs PC3 - Colored by Cancer Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Explained Variance
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_var) + 1), explained_var, marker='o', label='Individual Explained Variance')
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='s', label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by PCA Components')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("📉 Interpretation: Helps choose how many principal components to retain (e.g., first 5 for >80% variance).")

    # Clustering Dendrogram
    X_cluster = X_pca[:, :5]
    linkage_matrix = linkage(X_cluster, method='complete')

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=labels.values, leaf_rotation=90, leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
    plt.xlabel('Sample Index or Cancer Type')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    print("Interpretation: Shows hierarchical relationships and potential subgroups among cancer samples.")
    return X_scaled, labels

def evaluate_classification(X_scaled, labels):
    # Encode cancer types
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    # Binary classification for demonstration: cancer type 0 vs rest
    y_binary = (y_encoded == 0).astype(int)
    # Split and train a logistic regression
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Results (Binary Class: Type 0 vs Rest):")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Sensitivity (Recall):", sensitivity)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Interpretation: Even a simple logistic regression on standardized gene expression features can distinguish one cancer type from others to some extent.")

# Run the full workflow
X, labels = read_and_describe_data()
perform_eda(X, labels)
X_scaled, labels = apply_pca_and_clustering(X, labels)
evaluate_classification(X_scaled, labels)




#3 k means
import numpy as np
import matplotlib.pyplot as plt
def observation(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c='red', marker='o', s=50)
    plt.title('Observations')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
def cluster_labels(K, data):
    labels = np.random.choice(K, size=data.shape[0])
    return labels
def compute_centroids(K, data, labels):
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(K)])
    return centroids
def K_means_algorithm(data, K):
    labels = cluster_labels(K, data)
    centroids = compute_centroids(K, data, labels)
    while True:
        # (d) Assign each observation to the centroid to which it is closest
        new_labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        # (e) Repeat (c) and (d) until the answers obtained stop changing
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centroids = compute_centroids(K, data, labels)
    # (f) In your plot from (a), color the observations according to the cluster labels obtained
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='x', s=100, label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
def main():
    data = np.array([[1, 4], [1, 3], [0, 4], [5, 1], [6, 2], [4, 0]])
    K = 2
    observation(data)
    K_means_algorithm(data, K)
if __name__ == '__main__':
    main()

