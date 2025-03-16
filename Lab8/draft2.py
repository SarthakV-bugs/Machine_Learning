import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def lo_data():
    df = pd.read_csv("/ML_PRACTICE/Breast_cancer_wisconcsin.csv")
    x = df.drop(columns=["id", "diagnosis", "Unnamed: 32"])
    y = df["diagnosis"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def l2_norm(theta, lambda_value):
    return (np.sum(theta ** 2)) * lambda_value


def l1_norm(theta, lambda_value):
    return np.sum(np.abs(theta)) * lambda_value


def main():
    # Load the data
    x, y = lo_data()

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=999)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Range of lambda values to test
    lambda_values = np.logspace(-5, 2, 8)  # Test from 10^-5 to 10^2

    accuracy_lasso = []
    accuracy_ridge = []
    accuracy_no_penalty = []

    lasso_coeffs = []
    ridge_coeffs = []

    # Loop over different lambda values
    for lambda_value in lambda_values:
        print(f"Training with lambda: {lambda_value}")

        # Train a model with L1 regularization (Lasso)
        model_lasso = LogisticRegression(penalty='l1', C=1 / lambda_value, solver='liblinear')
        model_lasso.fit(x_train_scaled, y_train)
        y_pred_lasso = model_lasso.predict(x_test_scaled)

        # Train a model with L2 regularization (Ridge)
        model_ridge = LogisticRegression(penalty='l2', C=1 / lambda_value)
        model_ridge.fit(x_train_scaled, y_train)
        y_pred_ridge = model_ridge.predict(x_test_scaled)

        # Train a model without any penalty
        model_without_penalty = LogisticRegression(penalty=None)
        model_without_penalty.fit(x_train_scaled, y_train)
        y_pred_no_penalty = model_without_penalty.predict(x_test_scaled)

        # Get coefficients
        lasso_coeffs.append(model_lasso.coef_[0])
        ridge_coeffs.append(model_ridge.coef_[0])

        # Accuracy for each model
        accuracy_lasso.append(accuracy_score(y_test, y_pred_lasso))
        accuracy_ridge.append(accuracy_score(y_test, y_pred_ridge))
        accuracy_no_penalty.append(accuracy_score(y_test, y_pred_no_penalty))

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


if __name__ == '__main__':
    main()
