import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def load_data():

    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")


    x = df.drop(columns=["disease_score", "disease_score_fluct"]).to_numpy()
    y1 = df["disease_score"].to_numpy().reshape(-1, 1)
    y2 = df["disease_score_fluct"].to_numpy().reshape(-1, 1)

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
    x_mod, y1, _ = load_data()

    x_train, x_test, y_train, y_test = train_test_split(x_mod, y1, test_size=0.3, random_state=42)


    n_features = x_train.shape[1]
    theta = initialize_parameters(n_features)


    alpha = 0.01
    num_iterations = 1000


    costs = []


    for i in range(num_iterations):
        # Compute hypothesis
        h_t = hypothesis(x_train, theta)

        #cost
        cost_f = cost(h_t, y_train)
        costs.append(cost_f)

        #gradient
        grad_f = gradient(x_train, y_train, theta)

        #parameters
        theta = update_parameters(theta, grad_f, alpha)

        # Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Iter {i}, Cost: {cost_f}")

    # Final parameters and cost
    print("Final Parameters:\n", theta)
    final_cost = cost(hypothesis(x_test, theta), y_test)
    print(f"Final Test Cost: {final_cost}")


    #pred value
    y_predicted = hypothesis(x_test, theta)
    r2score = r2_score(y_test, y_predicted)
    print(f"R2score: {r2score}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(num_iterations), costs, color='blue', label="Cost")
    plt.title("Cost Function vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == '__main__':
    main()


