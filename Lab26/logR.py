import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("heart.csv")
X = df.drop('target', axis=1).values
y = df['target'].values

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, w):
    m = len(y)
    h = sigmoid(X @ w)
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient descent
def train_logistic(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    w = np.zeros(n + 1)
    for _ in range(epochs):
        h = sigmoid(X @ w)
        grad = (1/m) * X.T @ (h - y)
        w -= lr * grad
    return w

# Train
w = train_logistic(X_train, y_train)

# Predict
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
probs = sigmoid(X_test_bias @ w)
y_pred = (probs >= 0.5).astype(int)

# ROC and PR Curves
fpr, tpr, _ = roc_curve(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.title("ROC Curve")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.show()
