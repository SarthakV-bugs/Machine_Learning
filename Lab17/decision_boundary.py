# """Plotting decision boundaries using SVM"""
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn import svm
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.preprocessing import LabelEncoder
#
# # --- Data ---
# data = {
#     'x1': [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16],
#     'x2': [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3],
#     'Label': ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue',
#               'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red']
# }
# df = pd.DataFrame(data)
#
# # Extract features and labels
# x = df[['x1', 'x2']].to_numpy()
# y = df['Label'].to_numpy()
#
# # Encode labels to integers as labels are in string format
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)  # Blue -> 0, Red -> 1
#
# # --- Custom kernel and transform ---
# def transform(x1, x2):
#     return np.array([x1 ** 2, x2 ** 2, np.sqrt(2) * x1 * x2])
#
# def custom_kernel(X, Y):
#     X_mapped = np.array([transform(x[0], x[1]) for x in X])
#     Y_mapped = np.array([transform(y[0], y[1]) for y in Y])
#     return X_mapped @ Y_mapped.T  # dot product in transformed space
#
# # --- Plotting function ---
# def plot_decision_boundary(kernel=custom_kernel, ax=None, long_title=True, support_vectors=True):
#     clf = svm.SVC(kernel=kernel).fit(x, y_encoded)
#     print("Support vectors shape:", clf.support_vectors_.shape)
#     print("Support vectors :", clf.support_vectors_)
#     print("Support vectors indices :", clf.support_)
#     print("Support vectors (manually extracted):", x[clf.support_])
#
#     if ax is None:
#         _, ax = plt.subplots(figsize=(6, 5))
#     x_min, x_max = x[:, 0].min() - 2, x[:, 0].max() + 2
#     y_min, y_max = x[:, 1].min() - 2, x[:, 1].max() + 2
#     ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
#     # Plot decision boundary and margins
#     common_params = {"estimator": clf, "X": x, "ax": ax}
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="predict",
#         plot_method="pcolormesh",
#         alpha=0.3,
#         cmap='coolwarm'
#     )
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="decision_function",
#         plot_method="contour",
#         levels=[-1, 0, 1],
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#     )
#
#     # Plot support vectors
#     if support_vectors and clf.support_.shape[0] > 0:
#         sv = x[clf.support_]
#         ax.scatter(
#             sv[:, 0],
#             sv[:, 1],
#             s=150,
#             facecolors="none",
#             edgecolors="k",
#             label='Support Vectors'
#         )
#
#     # Plot data points
#     scatter = ax.scatter(x[:, 0], x[:, 1], c=y_encoded, cmap='coolwarm', s=40, edgecolors="k")
#     ax.legend(*scatter.legend_elements(), title="Classes")
#
#     # Set title
#     kernel_name = kernel.__name__ if callable(kernel) else str(kernel)
#     ax.set_title(f"Decision Boundary using {kernel_name} kernel" if long_title else kernel_name)
#
# # --- Plotting both kernels ---
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# plot_decision_boundary(kernel='poly', ax=axs[0])
# plot_decision_boundary(kernel=custom_kernel, ax=axs[1])
# plt.tight_layout()
# plt.show()
#
#





import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

# --- Data ---
data = {
    'x1': [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16],
    'x2': [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3],
    'Label': ['Blue'] * 8 + ['Red'] * 8
}
df = pd.DataFrame(data)

# Extract features and labels
x = df[['x1', 'x2']].to_numpy()
y = df['Label'].to_numpy()

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Custom kernel ---
def transform(x1, x2):
    return np.array([x1**2, x2**2, np.sqrt(2)*x1*x2])

def custom_kernel(X, Y):
    X_mapped = np.array([transform(x[0], x[1]) for x in X])
    Y_mapped = np.array([transform(y[0], y[1]) for y in Y])
    return X_mapped @ Y_mapped.T

# --- Plotting function ---
def plot_decision_boundary(kernel, ax, title, sv_label_tag):
    clf = svm.SVC(kernel=kernel).fit(x, y_encoded)

    print(f"Kernel: {title}")
    print("Support vectors shape:", clf.support_vectors_.shape)
    print("Support vectors:", clf.support_vectors_)
    print("Support vector indices:", clf.support_)
    print("Support vectors (from original x):", x[clf.support_])
    print("-" * 40)

    # Plot decision regions
    x_min, x_max = x[:, 0].min() - 2, x[:, 0].max() + 2
    y_min, y_max = x[:, 1].min() - 2, x[:, 1].max() + 2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    DecisionBoundaryDisplay.from_estimator(
        clf, x, ax=ax, response_method="predict",
        plot_method="pcolormesh", alpha=0.3, cmap='coolwarm'
    )
    DecisionBoundaryDisplay.from_estimator(
        clf, x, ax=ax, response_method="decision_function",
        plot_method="contour", levels=[-1, 0, 1],
        colors=["k", "k", "k"], linestyles=["--", "-", "--"]
    )

    # Plot support vectors with labels
    sv = x[clf.support_]
    ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none", edgecolors="k", linewidths=1.5, label=f'Support Vectors ({sv_label_tag})')
    for i, (x1, x2) in enumerate(sv):
        ax.text(x1 + 0.3, x2 + 0.3, f'SV ({sv_label_tag})', fontsize=8, color='black')

    # Plot original data
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y_encoded, cmap='coolwarm', s=40, edgecolors="k", label='Data Points')
    ax.legend(loc='upper right')
    ax.set_title(title)

# --- Plotting both kernels ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_boundary(kernel='poly', ax=axs[0], title='Polynomial Kernel (Built-in)', sv_label_tag='Poly')
plot_decision_boundary(kernel=custom_kernel, ax=axs[1], title='Custom Quadratic Kernel', sv_label_tag='Custom')
plt.suptitle("SVM Decision Boundaries with Support Vectors Labeled by Kernel", fontsize=16, y=1.03)
plt.tight_layout()
plt.show()

