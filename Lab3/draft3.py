import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def load_data():
    # Read the CSV file
    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")

    # Extract features and target variables
    x = df.drop(columns=["disease_score", "disease_score_fluct"]).to_numpy()
    y1 = df["disease_score"].to_numpy().reshape(-1, 1)  # Target variable 1 as a column vector
    y2 = df["disease_score_fluct"].to_numpy().reshape(-1, 1)  # Target variable 2 as a column vector

    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
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


def normal_equation(x_mod, y1,y2):

   nor_eq_theta = np.linalg.inv(x_mod.T.dot(x_mod)).dot(x_mod.T.dot(y2))
   return nor_eq_theta


def main():
    # Load data
    x_mod, y1, y2 = load_data()

    # load data

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x_mod, y2, test_size=0.3, random_state=42)

    #each data into a csv file
    pd.DataFrame(x_train).to_csv('x_train.csv', index=False, header=False)
    pd.DataFrame(x_test).to_csv('x_test.csv', index=False, header=False)
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False, header=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False, header=False)


    # x_train, x_test, y_train, y_test = train_test_split(x_mod, y2, test_size=0.3, random_state=999)

    # Initialize parameters
    n_features = x_train.shape[1]
    theta = initialize_parameters(n_features)

    alpha = 0.01  # Learning rate
    num_iterations = 1000

    # costs for plotting
    costs = []

    #normal equation from this model
    print(normal_equation(x_mod, y1, y2))

    # [[8.13355133e+02]
    #  [4.30217047e+01]
    #  [3.79453222e+00]
    #  [1.50563560e+01]
    #  [1.53895416e+01]
    #  [4.99767458e-01]]

    # sciklearn optimized thetas
    # model = LinearRegression()
    # model.fit(x_train, y_train)
    # y_pred_sklearn = model.predict(x_mod)
    # r2_sklearn = r2_score(y_test, y_pred_sklearn)
    # print(f"R2 Score (Scikit-learn): {r2_sklearn}")

    # coefficients, [[ 0.         44.71679394  2.57532146 23.61439022  9.5935897  -5.57520727]]



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
    print(f"y_test: {y_test}")
    print(f"Final Test Cost: {final_cost}")

    #using GDA
    # y_pred_gd = hypothesis(x_test, theta)
    y_pred_gd = hypothesis(x_mod, theta)
    print(f"y_pred: {y_pred_gd}")


    # Calculate R2 score using sklearn's r2_score
    # y_pred_sk = model.predict(x_test)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred_sk = model.predict(x_mod)
    r2 = r2_score(y2, y_pred_sk)
    # print(f"R2 Score: {r2}")

    # plot b/w GDA and True value
    # plt.subplot(1, 3, 1)
    # plt.scatter(y_test, y_pred_gd, color='blue', label='Gradient Descent')
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    # plt.title("Gradient Descent: Predicted vs True")
    # plt.xlabel("True Disease Fluctuation")
    # plt.ylabel("Predicted Disease Fluctuation")
    # plt.legend()

    #plot Y_pred_test and x_test



    # Plot cost vs. iterations
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(num_iterations), costs, color='blue', label="Cost")
    # plt.title("Cost Function vs. Iterations")
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.grid()
    # plt.show()

    #plot normal equation, gradient descent and Scikit learn for disease fluctuation.

    # print(x_test)
    # print(x_test[:,1])
    # print(y_pred_gd.shape)



    # Normal Equation Predictions
    theta_normal = normal_equation(x_mod, y1, y2)
    # y_pred_normal = hypothesis(x_test, theta_normal)
    y_pred_normal = hypothesis(x_mod, theta_normal)


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_mod[:,1],y2,color='blue',label='Actual Data')
    plt.scatter(x_mod[:,1], y_pred_gd, color='yellow', label='Gradient Descent', linewidth=2)
    plt.scatter(x_mod[:,1], y_pred_sk, color='red', label='Scikit-learn',linewidth=2)
    plt.scatter(x_mod[:,1], y_pred_normal, color='green',  label='Normal Equation',linewidth=2)

    # Graph aesthetics
    plt.title("Predicted Disease Fluctuation vs Age Feature")
    plt.xlabel("Age Feature (x_test[:, 1])")
    plt.ylabel("Predicted Disease Fluctuation")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
