import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load Breast Cancer dataset
def lo_data():
    df = pd.read_csv("/ML_PRACTICE/Breast_cancer_wisconcsin.csv")
    x = df.drop(columns=["id", "diagnosis", "Unnamed: 32"])  # Drop non-features columns
    y = df["diagnosis"]
    le = LabelEncoder()
    y = le.fit_transform(y)  # Encode labels (M=1, B=0)
    return x, y


# L2 Norm (Ridge Regularization) function
def l2_norm(theta, lambda_value):
    return (np.sum(theta ** 2)) * lambda_value


# L1 Norm (Lasso Regularization) function
def l1_norm(theta, lambda_value):
    return np.sum(np.abs(theta)) * lambda_value


# Main function
def main():
    # Load the data
    x, y = lo_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=999)

    # Standardize the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Range of lambda values to test
    lambda_values = np.logspace(-5, 2, 8)  # Test from 10^-5 to 10^2

    # Lists to store results for different regularization techniques
    accuracy_lasso = []
    accuracy_ridge = []
    accuracy_no_penalty = []

    lasso_coeffs = []
    ridge_coeffs = []

    # Variables to track the best accuracy and corresponding lambda for each model
    best_accuracy_lasso = -np.inf
    best_lambda_lasso = None
    best_accuracy_ridge = -np.inf
    best_lambda_ridge = None
    best_accuracy_no_penalty = -np.inf

    # Loop over different lambda values for Lasso, Ridge, and No Penalty models
    for lambda_value in lambda_values:
        print(f"\nTraining with lambda: {lambda_value}")

        # Train a model with L1 regularization (Lasso)
        model_lasso = LogisticRegression(penalty='l1', C=1 / lambda_value, solver='liblinear')
        model_lasso.fit(x_train_scaled, y_train)
        y_pred_lasso = model_lasso.predict(x_test_scaled)

        # Calculate and print accuracy for Lasso
        accuracy_lasso_score = accuracy_score(y_test, y_pred_lasso)
        accuracy_lasso.append(accuracy_lasso_score)
        print(f"Lasso Accuracy: {accuracy_lasso_score:.4f}")

        # Update best accuracy for Lasso
        if accuracy_lasso_score > best_accuracy_lasso:
            best_accuracy_lasso = accuracy_lasso_score
            best_lambda_lasso = lambda_value

        # Train a model with L2 regularization (Ridge)
        model_ridge = LogisticRegression(penalty='l2', C=1 / lambda_value)
        model_ridge.fit(x_train_scaled, y_train)
        y_pred_ridge = model_ridge.predict(x_test_scaled)

        # Calculate and print accuracy for Ridge
        accuracy_ridge_score = accuracy_score(y_test, y_pred_ridge)
        accuracy_ridge.append(accuracy_ridge_score)
        print(f"Ridge Accuracy: {accuracy_ridge_score:.4f}")

        # Update best accuracy for Ridge
        if accuracy_ridge_score > best_accuracy_ridge:
            best_accuracy_ridge = accuracy_ridge_score
            best_lambda_ridge = lambda_value

        # Train a model without any regularization (No Penalty)
        model_without_penalty = LogisticRegression(penalty=None)
        model_without_penalty.fit(x_train_scaled, y_train)
        y_pred_no_penalty = model_without_penalty.predict(x_test_scaled)

        # Calculate and print accuracy for No Penalty
        accuracy_no_penalty_score = accuracy_score(y_test, y_pred_no_penalty)
        accuracy_no_penalty.append(accuracy_no_penalty_score)
        print(f"No Penalty Accuracy: {accuracy_no_penalty_score:.4f}")

        # Update best accuracy for No Penalty
        if accuracy_no_penalty_score > best_accuracy_no_penalty:
            best_accuracy_no_penalty = accuracy_no_penalty_score

        # Get coefficients for each model
        lasso_coeffs.append(model_lasso.coef_[0])
        ridge_coeffs.append(model_ridge.coef_[0])

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

    # Print the best accuracy and corresponding lambda for each model
    print(f"\nBest Accuracy for Lasso: {best_accuracy_lasso:.4f} at lambda = {best_lambda_lasso}")
    print(f"Best Accuracy for Ridge: {best_accuracy_ridge:.4f} at lambda = {best_lambda_ridge}")
    print(f"Best Accuracy for No Penalty: {best_accuracy_no_penalty:.4f}")


# Run the main function
if __name__ == '__main__':
    main()
