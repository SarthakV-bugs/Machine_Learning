import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load data
def load_data():
    data = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/sonar.csv/sonar.csv", header=None)
    column_names = [f"Frequency{i}" for i in range(1, data.shape[1])] + ["Class"]
    data.columns = column_names
    y = data["Class"].map({'M': 1, 'R': 0})  # Map target variable to binary
    X = data.drop("Class", axis=1)
    return X, y


# Standardization (manual method)
def standardize_manual(X, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


# Logistic Regression with k-fold cross-validation
def logistic_regression_k_folds():
    X, y = load_data()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Range of lambda values to test
    lambda_values = np.logspace(-5, 2, 8)  # Test from 10^-5 to 10^2

    accuracy_lasso = []
    accuracy_ridge = []
    accuracy_no_penalty = []

    lasso_coeffs = []
    ridge_coeffs = []

    best_lambda_lasso = None
    best_accuracy_lasso = -np.inf
    best_lambda_ridge = None
    best_accuracy_ridge = -np.inf
    best_accuracy_no_penalty = -np.inf

    # Loop over different lambda values
    for lambda_value in lambda_values:
        print(f"Training with lambda: {lambda_value}")

        # 1. Logistic Regression with L1 regularization (Lasso)
        model_lasso = LogisticRegression(penalty='l1', C=1 / lambda_value, solver='liblinear', max_iter=3000)
        scores_lasso = cross_val_score(model_lasso, X, y, cv=cv, scoring="accuracy")
        accuracy_lasso.append(np.mean(scores_lasso))
        lasso_model = model_lasso.fit(X, y)
        lasso_coeffs.append(lasso_model.coef_[0])

        if np.mean(scores_lasso) > best_accuracy_lasso:
            best_accuracy_lasso = np.mean(scores_lasso)
            best_lambda_lasso = lambda_value

        # 2. Logistic Regression with L2 regularization (Ridge)
        model_ridge = LogisticRegression(penalty='l2', C=1 / lambda_value, max_iter=3000)
        scores_ridge = cross_val_score(model_ridge, X, y, cv=cv, scoring="accuracy")
        accuracy_ridge.append(np.mean(scores_ridge))
        ridge_model = model_ridge.fit(X, y)
        ridge_coeffs.append(ridge_model.coef_[0])

        if np.mean(scores_ridge) > best_accuracy_ridge:
            best_accuracy_ridge = np.mean(scores_ridge)
            best_lambda_ridge = lambda_value

        # 3. Logistic Regression without penalty
        model_no_penalty = LogisticRegression(penalty=None, max_iter=3000)
        scores_no_penalty = cross_val_score(model_no_penalty, X, y, cv=cv, scoring="accuracy")
        accuracy_no_penalty.append(np.mean(scores_no_penalty))

        if np.mean(scores_no_penalty) > best_accuracy_no_penalty:
            best_accuracy_no_penalty = np.mean(scores_no_penalty)

        # Print Coefficients for Lasso, Ridge, and No Penalty models
        print(f"Lambda: {lambda_value}")
        print(f"  Lasso Coefficients: {lasso_model.coef_[0]}")
        print(f"  Ridge Coefficients: {ridge_model.coef_[0]}")
        print(f"  No Penalty Coefficients: {model_no_penalty.fit(X, y).coef_[0]}")

        # Identify and print features with zero coefficients in Lasso model
        zero_coeff_features = np.where(lasso_model.coef_[0] == 0)[0]
        if len(zero_coeff_features) > 0:
            print("  Features set to zero in Lasso:")
            for feature in zero_coeff_features:
                print(f"    {X.columns[feature]}")

    # Print the best lambda and accuracy for each model
    print(f"\nBest Lambda for Lasso: {best_lambda_lasso} with Accuracy: {best_accuracy_lasso:.4f}")
    print(f"Best Lambda for Ridge: {best_lambda_ridge} with Accuracy: {best_accuracy_ridge:.4f}")
    print(f"Best Accuracy for No Penalty: {best_accuracy_no_penalty:.4f}")

    # Plotting accuracy comparison for different lambda values
    plt.figure(figsize=(6, 4))
    plt.plot(np.log10(lambda_values), np.array(accuracy_lasso) * 100, label='Lasso (L1)', marker='o')
    plt.plot(np.log10(lambda_values), np.array(accuracy_ridge) * 100, label='Ridge (L2)', marker='o')
    plt.plot(np.log10(lambda_values), np.array(accuracy_no_penalty) * 100, label='No Penalty', marker='o')
    plt.xlabel('log10(lambda)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Regularization Strength (Lambda)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the coefficients for Lasso and Ridge for different lambda values
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    for i in range(len(lasso_coeffs[0])):
        plt.plot(np.log10(lambda_values), [coeff[i] for coeff in lasso_coeffs], label=f'Feature {i + 1}')
    plt.title('Lasso (L1 Regularization) Coefficients')
    plt.xlabel('log10(lambda)')
    plt.ylabel('Coefficient Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(len(ridge_coeffs[0])):
        plt.plot(np.log10(lambda_values), [coeff[i] for coeff in ridge_coeffs], label=f'Feature {i + 1}')
    plt.title('Ridge (L2 Regularization) Coefficients')
    plt.xlabel('log10(lambda)')
    plt.ylabel('Coefficient Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logistic_regression_k_folds()
