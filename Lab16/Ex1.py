import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.datasets import make_regression

def load_data():
    X, y = make_regression(n_samples=500, n_features=5, noise=15, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_trees(X_train, y_train, total_trees=10):
    forest = []
    for _ in range(total_trees):
        # Pick a random sample of the training data (with replacement)
        X_sample, y_sample = resample(X_train, y_train)
        # Train a small decision tree on that sample
        tree = DecisionTreeRegressor(max_depth=5)
        tree.fit(X_sample, y_sample)
        forest.append(tree)
    return forest


def predictions(trees, X_test):
    all_preds = []
    for tree in trees:
        preds = tree.predict(X_test)
        all_preds.append(preds)
    return np.mean(all_preds, axis=0)


def main():
    X_train, X_test, y_train, y_test = load_data()
    trees = train_trees(X_train, y_train, total_trees=10)
    pred = predictions(trees, X_test)
    error = mean_squared_error(y_test, pred)
    print(f"Mean Squared Error: {error:.2f}")

if __name__ == "__main__":
    main()
