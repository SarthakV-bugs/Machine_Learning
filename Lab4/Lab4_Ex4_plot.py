import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def load_data():
    # Read the CSV file
    df = pd.read_csv(r"/simulated_data_multiple_linear_regression_for_ML.csv")

    # Extract features and target variables
    x = df["age"].values.reshape(-1, 1)
    y1 = df["disease_score"].to_numpy().reshape(-1, 1)  # Target variable 1 as a column vector
    y2 = df["disease_score_fluct"].to_numpy().reshape(-1, 1)  # Target variable 2 as a column vector

    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = np.c_[np.ones(x.shape[0]), x]  # Add a column of ones for the bias term

    return x, y1, y2


def initialize_parameters(n_features):
    np.random.seed(42)
    theta = np.random.uniform(low=-0.1, high=0.1, size=(n_features, 1))  # Ensure theta is a column vector
    return theta


def hypothesis(x, theta):
    return x.dot(theta)  # Matrix multiplication


def cost(h_t, y):
    m = len(y)
    loss = h_t - y
    cost_f = (1 / (2 * m)) * np.sum(loss ** 2)
    return cost_f


def gradient(x, y, theta):
    m = len(y)
    grad_f = (1 / m) * x.T.dot(x.dot(theta) - y)
    return grad_f


def update_parameters(theta, grad_f, alpha):
    new_theta = theta - alpha * grad_f
    return new_theta


def normal_equation(x, y1, y2):
    # Using normal equation to solve for theta
    nor_eq_theta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y2))
    return nor_eq_theta


def main():
    # Load data
    x, y1, y2 = load_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.3, random_state=42)

    # Save the data into CSV files
    pd.DataFrame(x_train).to_csv('../x_train.csv', index=False, header=False)
    pd.DataFrame(x_test).to_csv('../x_test.csv', index=False, header=False)
    pd.DataFrame(y_train).to_csv('../y_train.csv', index=False, header=False)
    pd.DataFrame(y_test).to_csv('../y_test.csv', index=False, header=False)

    # Initialize parameters
    n_features = x_train.shape[1]  # Number of features (including the bias term)
    theta = initialize_parameters(n_features)  # Initialize theta as a column vector

    alpha = 0.01  # Learning rate
    num_iterations = 1000

    # Costs for plotting
    costs = []

    # Normal equation to get the theta values
    print("Normal Equation Solution:\n", normal_equation(x, y1, y2))

    # Gradient descent loop
    for i in range(num_iterations):
        # Compute hypothesis
        h_t = hypothesis(x_train, theta)

        # Compute cost
        cost_f = cost(h_t, y_train)
        costs.append(cost_f)

        # Compute gradient
        grad_f = gradient(x_train, y_train, theta)

        # Update parameters
        theta = update_parameters(theta, grad_f, alpha)

        # Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Iter {i}, Cost: {cost_f}")

    # Final parameters and cost
    print("Final Parameters:\n", theta)
    final_cost = cost(hypothesis(x_test, theta), y_test)
    print(f"Final Test Cost: {final_cost}")

    # Using Gradient Descent predictions
    y_pred_gd = hypothesis(x, theta)
    print(f"y_pred (Gradient Descent): {y_pred_gd}")

    # Calculate R2 score using sklearn's LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred_sk = model.predict(x)
    r2 = r2_score(y2, y_pred_sk)
    print(f"R2 Score (Scikit-learn): {r2}")
    print()

    # Normal Equation Predictions
    theta_normal = normal_equation(x, y1, y2)
    y_pred_normal = hypothesis(x, theta_normal)  # Using the normal equation for predictions
    print()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Scatter plot of the actual data vs. predicted values
    plt.scatter(x[:, 1], y2, color='blue', label='Actual Data')  # Actual data for y2

    # Use x_test for predictions from Gradient Descent, Scikit-learn, and Normal Equation
    plt.plot(x[:, 1], y_pred_gd, color='yellow', label='Gradient Descent', linewidth=1)
    plt.plot(x[:, 1], y_pred_sk, color='red', label='Scikit-learn', linewidth=1)
    plt.plot(x[:, 1], y_pred_normal, color='green', label='Normal Equation', linewidth=1)


    plt.title("Predicted Disease Fluctuation vs Age Feature")
    plt.xlabel("Age Feature (x_test[:, 1])")
    plt.ylabel("Predicted Disease Fluctuation")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()