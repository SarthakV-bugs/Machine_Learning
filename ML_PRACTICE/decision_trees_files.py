

######Clean Code for both classification and regression setting
import numpy as np
import pandas as pd
from math import inf
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split


# Node class for decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Split threshold
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted value (for leaves)


# Function to compute unique midpoints for potential split thresholds
def compute_thresholds(x):
    thresholds = {}
    for feature in x.columns:
        unique_values = sorted(x[feature].unique())
        midpoints = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]
        thresholds[feature] = midpoints
    return thresholds


# Function to compute entropy for classification
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


# Function to compute information gain for classification
def information_gain(y, left_y, right_y):
    parent_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)
    weight_left = len(left_y) / len(y)
    weight_right = len(right_y) / len(y)
    return parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)


# Function to find the best split for classification
def best_split(x, y, thresholds):
    best_feature, best_threshold, best_ig = None, None, -inf
    for feature in x.columns:
        for threshold in thresholds[feature]:
            left_mask = x[feature] <= threshold
            right_mask = x[feature] > threshold
            left_y, right_y = y[left_mask], y[right_mask]
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            ig = information_gain(y, left_y, right_y)
            if ig > best_ig:
                best_feature, best_threshold, best_ig = feature, threshold, ig
    return best_feature, best_threshold, best_ig


# Function to build the decision tree recursively for classification
def build_tree(x, y, thresholds, depth=0, max_depth=None):
    if len(set(y)) == 1:
        return Node(value=y.iloc[0])
    if max_depth is not None and depth >= max_depth:
        return Node(value=y.mode()[0])
    best_feature, best_threshold, best_ig = best_split(x, y, thresholds)
    if best_ig <= 0:
        return Node(value=y.mode()[0])
    left_mask = x[best_feature] <= best_threshold
    right_mask = x[best_feature] > best_threshold
    left_subtree = build_tree(x[left_mask], y[left_mask], thresholds, depth + 1, max_depth)
    right_subtree = build_tree(x[right_mask], y[right_mask], thresholds, depth + 1, max_depth)
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)


# Prediction function
def predict(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)


# Accuracy calculation
def accuracy_score(y_true, y_pred):
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)


# Train-test evaluation
def train_test_eval(x, y, test_size=0.3, max_depth=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    thresholds = compute_thresholds(x_train)
    tree = build_tree(x_train, y_train, thresholds, max_depth=max_depth)
    y_pred = [predict(tree, x_test.iloc[i]) for i in range(len(x_test))]
    return accuracy_score(y_test, y_pred)


# Main function for classification
def classification_main():
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    accuracy = train_test_eval(x, y, max_depth=5)
    print(f"Decision Tree Classifier Accuracy: {accuracy:.2f}")


# Main function for regression using MSE
def regression_main():
    diabetes = load_diabetes()
    x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)

    def mse(y):
        return np.mean((y - y.mean()) ** 2)

    def mse_reduction(y, left_y, right_y):
        return mse(y) - (len(left_y) / len(y) * mse(left_y) + len(right_y) / len(y) * mse(right_y))

    def best_split_regression(x, y, thresholds):
        best_feature, best_threshold, best_mse_red = None, None, -inf
        for feature in x.columns:
            for threshold in thresholds[feature]:
                left_mask = x[feature] <= threshold
                right_mask = x[feature] > threshold
                left_y, right_y = y[left_mask], y[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                mse_red = mse_reduction(y, left_y, right_y)
                if mse_red > best_mse_red:
                    best_feature, best_threshold, best_mse_red = feature, threshold, mse_red
        return best_feature, best_threshold, best_mse_red

    def build_tree_regression(x, y, thresholds, depth=0, max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return Node(value=y.mean())
        best_feature, best_threshold, best_mse_red = best_split_regression(x, y, thresholds)
        if best_mse_red <= 0:
            return Node(value=y.mean())
        left_mask = x[best_feature] <= best_threshold
        right_mask = x[best_feature] > best_threshold
        left_subtree = build_tree_regression(x[left_mask], y[left_mask], thresholds, depth + 1, max_depth)
        right_subtree = build_tree_regression(x[right_mask], y[right_mask], thresholds, depth + 1, max_depth)
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    thresholds = compute_thresholds(x_train)
    tree = build_tree_regression(x_train, y_train, thresholds, max_depth=5)
    y_pred = [predict(tree, x_test.iloc[i]) for i in range(len(x_test))]
    mse_error = mse(y_test - y_pred)
    print(f"Decision Tree Regression MSE: {mse_error:.2f}")


if __name__ == "__main__":
    classification_main()
    regression_main()
#
