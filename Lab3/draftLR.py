import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():

    # Read the CSV file
    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")

    # Extract features and target variables
    x = df.drop(columns=["disease_score", "disease_score_fluct"]).to_numpy()
    y1 = df["disease_score"].to_numpy().reshape(-1, 1)  # Target variable 1 as a column vector
    y2 = df["disease_score_fluct"].to_numpy().reshape(-1, 1)  # Target variable 2 as a column vector

    # Normalize features
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Add a column of ones for the intercept term
    x_mod = np.c_[np.ones(x.shape[0]), x]

    return x_mod, y1, y2


def initialize_parameters(n_features):

    np.random.seed(42)
    theta = np.random.uniform(low=-0.1, high=0.1, size=n_features)
    return theta.reshape(-1, 1)


def hypothesis(x_mod, theta):

    return x_mod.dot(theta)


def cost(h_t, y):

    m = len(y)
    loss = h_t - y
    cost_f = (1 / (2 * m)) * np.sum(loss ** 2)
    return cost_f


def gradient(x_mod, y, theta):
    m = len(y)
    grad_f = (1 / m) * x_mod.T.dot(x_mod.dot(theta) - y)
    return grad_f


def update_parameters(theta, grad_f, alpha):

    new_theta = theta - alpha * grad_f
    return new_theta


def main():
    # Load data
    x_mod, y1, _ = load_data()

    # Initialize parameters
    n_features = x_mod.shape[1]
    theta = initialize_parameters(n_features)

    # Set hyperparameters
    alpha = 0.01  # Learning rate
    num_iterations = 10000

    #costs
    costs = []

    # Gradient descent loop
    for i in range(num_iterations):
        # Compute hypothesis
        h_t = hypothesis(x_mod, theta)

        # Compute cost
        cost_f = cost(h_t, y1)
        costs.append(cost_f)

        # Compute gradient
        grad_f = gradient(x_mod, y1, theta)

        # Update parameters
        theta = update_parameters(theta, grad_f, alpha)

        # Print cost every 100 iterations
        if i % 10 == 0:
            print(f"Iter {i}, Cost: {cost_f}")

    # Final parameters and cost
    print("Final Parameters:\n", theta)
    final_cost = cost(hypothesis(x_mod, theta), y1)
    print(f"Final Cost: {final_cost}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(num_iterations), costs, color='blue',  label="Cost")
    plt.title("Cost Function vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
