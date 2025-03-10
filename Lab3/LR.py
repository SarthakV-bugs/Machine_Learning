import pandas as pd
import numpy as np


def load_data():

    # read the csv files
    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")

    # Form x and y
    x = df.drop(columns=["disease_score","disease_score_fluct"]).to_numpy()
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0) #normalize the data
    x_mod = np.c_[np.ones(x.shape[0]), x] #x with ones column

    # print(x_mod)

    y1 = df["disease_score"]
    y1_vector = y1.to_numpy().reshape(60,1)

    y2 = df["disease_score_fluct"]
    y2_vector = y2.to_numpy()
    # print(y2_vector)

    return x, x_mod,y1_vector,y2_vector

def hypo():

    x, x_mod , y1_vector, y2_vector = load_data()

    np.random.seed(42)

    theta = np.random.uniform(low=0, high=0.1, size=x.shape[1])
    theta0 = np.random.uniform(low=0, high=0.01)
    theta_mod = np.insert(theta,0,theta0)   #theta vector with theta0

    print(theta_mod)
    print(theta_mod.shape)

    #hyptothesis
    h_t = x_mod.dot(theta_mod.T)
    h_t_reshaped = h_t.reshape(60,1)


    return h_t_reshaped, theta_mod, x_mod


def cost(h_t_reshaped,y1_vector):

    loss_f = h_t_reshaped - y1_vector
    loss_f_trans = loss_f.T
    cost_f = (loss_f_trans.dot(loss_f))/2
    return cost_f

def gradient(x_mod, y1_vector, theta_mod):

    x_t_x = x_mod.T.dot(x_mod)
    print(x_t_x)
    theta_mod_reshape = theta_mod.reshape(-1,1)
    print(theta_mod_reshape.shape)

    grad_f = x_t_x.dot(theta_mod_reshape)-(x_mod.T).dot(y1_vector) #for 6 theta values
    return grad_f

def update_parameters(theta_mod, alpha, grad_f):

    theta_mod = theta_mod - alpha * grad_f
    return theta_mod



def main():
    x, x_mod, y1_vector, y2_vector = load_data()
    h_t_reshaped, theta_mod, x_mod = hypo()

    # print(cost(h_t_reshaped,y1_vector))

    print(gradient(x_mod,y1_vector, theta_mod))

    alpha = 0.01
    iterations = 1000
    for i in range(iterations):
        grad_f = gradient(x_mod, y1_vector, theta_mod)

        cost_f = cost(h_t_reshaped,y1_vector)
        theta_mod = theta_mod - alpha * grad_f.flatten()

        h_t_reshaped = x_mod.dot(theta_mod.T).reshape(-1, 1)

        if i % 100:
            cost_new = cost_f(h_t_reshaped, theta_mod)
            print(f"cost new, {cost_new}")







if __name__ == '__main__':
    main()