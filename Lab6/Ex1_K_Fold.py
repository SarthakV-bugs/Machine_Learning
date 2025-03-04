import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
# Data normalization - scale the values between 0 and 1. Implement code from scratch.
# Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
# Use validation set to do feature and model selection.


def load_data(filepath, target_column, columns_to_drop):
    df = pd.read_csv(filepath)
    x = df.drop(columns=columns_to_drop).values
    y = df[target_column].values


    x = np.c_[np.ones(x.shape[0]), x]
    return x, y


def k_fold_split(x, y, k=10):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    fold_size = len(x) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(x)
        val_indices = indices[start:end]
        train_indices = np.setdiff1d(indices, val_indices)
        folds.append((train_indices, val_indices))

    return folds


def normalize(x_train, x_val):
    mean = np.mean(x_train[:, 1:], axis=0)
    std = np.std(x_train[:, 1:], axis=0)

    x_train_norm = x_train.copy()
    x_train_norm[:, 1:] = (x_train[:, 1:] - mean) / std

    x_val_norm = x_val.copy()
    x_val_norm[:, 1:] = (x_val[:, 1:] - mean) / std

    return x_train_norm, x_val_norm


def cross_validate(x, y, k=10, alpha=0.01, iterations=1000):
    folds = k_fold_split(x, y, k)
    r2_scores = []

    for train_idx, val_idx in folds:
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalize using training data stats
        x_train_norm, x_val_norm = normalize(x_train, x_val)

        # Train model
        theta = np.random.rand(x_train_norm.shape[1]) * 0.1
        cost_history = []

        for _ in range(iterations):
            # Calculate predictions and error
            predictions = x_train_norm @ theta
            error = predictions - y_train

            # Update parameters
            gradients = (x_train_norm.T @ error) / len(y_train)
            theta -= alpha * gradients

            # Calculate cost
            cost = np.mean(error ** 2)
            cost_history.append(cost)

            if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < 1e-5:
                break

        # Validation
        val_pred = x_val_norm @ theta
        r2 = r2_score(y_val, val_pred)
        r2_scores.append(r2)

    return np.mean(r2_scores)


def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


# Usage
if __name__ == "__main__":
    x, y = load_data("/home/ibab/PycharmProjects/ML-Lab/Lab4/simulated_data_multiple_linear_regression_for_ML.csv",
                     "disease_score_fluct",
                     ["disease_score", "disease_score_fluct"])

    avg_r2 = cross_validate(x, y, k=10)
    print(f"Average R² score (5-fold CV): {avg_r2:.4f}")
