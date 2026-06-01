import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(42)

# Generate 2-class moon-shaped data
X, y = make_moons(n_samples=200, noise=0.3)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create DataFrame for plotting
df_train = pd.DataFrame(X_train, columns=["x1", "x2"])
df_train["y"] = y_train
df_test = pd.DataFrame(X_test, columns=["x1", "x2"])
df_test["y"] = y_test


# Helper function to plot decision boundary
def plot_decision_boundary(model, X, y, title="", show_support_vectors=False):
    x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
    x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    x_grid = np.c_[x1.ravel(), x2.ravel()]

    decision = model.decision_function(x_grid)
    decision = decision.reshape(x1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(x1, x2, decision, levels=20, cmap='coolwarm', alpha=0.5)
    plt.contour(x1, x2, decision, levels=[0], linewidths=2, colors='k')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', s=60, edgecolor='k')

    if show_support_vectors and hasattr(model, 'support_vectors_'):
        sv = model.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='black', label='Support Vectors')

    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Train and evaluate three SVMs
kernels = {
    "Linear SVM": SVC(kernel='linear'),
    "Polynomial SVM (deg=3)": SVC(kernel='poly', degree=3),
    "RBF SVM": SVC(kernel='rbf')
}

for name, model in kernels.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"{name}")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy:  {test_acc:.3f}\n")

    plot_decision_boundary(model, X_train, y_train, title=f"{name} - Train", show_support_vectors=True)
    plot_decision_boundary(model, X_test, y_test, title=f"{name} - Test", show_support_vectors=True)
