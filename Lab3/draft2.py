import pandas as pd
import numpy as np


def load_data():

    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")

    # Form x and y
    x = df.drop(columns=["disease_score", "disease_score_fluct"]).to_numpy()
    # x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)  # Normalize the data
    x_mod = np.c_[np.ones(x.shape[0]), x]  # Add a column of ones for bias term

    y1 = df["disease_score"].to_numpy().reshape(-1, 1)
    y2 = df["disease_score_fluct"].to_numpy()

    return x, x_mod, y1, y2


def hypo(x_mod, theta_mod):
    return x_mod.dot(theta_mod.T).reshape(-1, 1)


def cost(h_t_reshaped, y1_vector):
    loss_f = h_t_reshaped - y1_vector
    cost_f = (loss_f.T.dot(loss_f)) / (2 * len(y1_vector))
    return cost_f[0, 0]


def gradient(x_mod, y1_vector, theta_mod):

    grad_f = (x_mod.T.dot(x_mod.dot(theta_mod.T).reshape(-1, 1) - y1_vector)) / len(y1_vector)
    return grad_f.flatten()


def main():
    # Load data
    x, x_mod, y1_vector, y2_vector = load_data()

    # Initialize parameters
    np.random.seed(42)
    theta = np.random.uniform(low=0, high=0.1, size=x.shape[1])
    theta0 = np.random.uniform(low=0, high=0.01)
    theta_mod = np.insert(theta, 0, theta0)  # Add bias term

    alpha = 0.01  # Learning rate
    iterations = 1000

    for i in range(iterations):
        # Compute hypothesis
        h_t_reshaped = hypo(x_mod, theta_mod)

        # Compute gradient
        grad_f = gradient(x_mod, y1_vector, theta_mod)

        # Update parameters
        theta_mod = theta_mod - alpha * grad_f

        # Compute cost every 100 iterations
        if i % 100 == 0:
            cost_f = cost(h_t_reshaped, y1_vector)
            print(f"Iteration {i}, Cost: {cost_f}")

    print("Final Parameters:", theta_mod)


if __name__ == "__main__":
    main()
