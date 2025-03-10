# Read simulated data csv file
# Form x and y
# Write a function to compute hypothesis
# Write a function to compute the cost
# Write a function to compute the derivative
# Write update parameters logic in the main function


import pandas as pd
import numpy as np
from random import uniform



def load_data():

    # read the csv files
    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")

    # Form x and y
    x = df.drop(columns=["disease_score","disease_score_fluct"])#X features
    y1 = df["disease_score"]
    y2 = df["disease_score_fluct"]

    return x,y1,y2

# def hypothesis function
def hyp():

    x, y1, y2 = load_data()

    np.random.seed(42)
    theta = np.random.uniform(low=0,high=0.1,size=x.shape[1])
    print(theta)
    theta0 = np.random.uniform(low=0,high=0.01)
    x0 = 1

    mod_x = np.c_[np.ones(x.shape[0]), x]
    # print(mod_x.shape) #adding ones column to the frame
    # mod_theta = np.append(theta, theta0)
    mod_theta = np.insert(theta,0,theta0)
    print(mod_theta)
    # print(mod_theta.shape)



    h_theta = mod_x.dot(mod_theta.T)
    h_theta_reshaped = h_theta.reshape(60,1)
    print(type(h_theta_reshaped))
    return h_theta_reshaped,mod_theta,mod_x

#
# def cost(h_theta_reshaped,y1):
#
#     loss_f = h_theta_reshaped - (y1.to_numpy().reshape(60, 1))
#     loss_f_trans = loss_f.T
#     cost_f = (loss_f_trans.dot(loss_f))/2
#     return cost_f

#
# def gradient(mod_x,mod_theta,y1):
#
#     y1_reshaped = y1.to_numpy().reshape(-1,1)
#     print(y1_reshaped.shape)
#
#     mod_the_x = mod_x.dot(mod_theta)
#     print(mod_the_x.shape)
#     # mod_the_x = mod_the_x.reshape(-1,1)
#     grad_f = mod_x.T.dot(mod_the_x) - mod_x.T.dot(y1_reshaped)
#
#     return grad_f

def main():
    x,y1,y2= load_data()
    print(type(y1))

    h_theta_reshaped, mod_x,mod_theta = hyp()

    # print(cost(h_theta_reshaped,y1))

    y1_reshaped = y1.to_numpy().reshape(-1, 1)
    print(y1_reshaped.shape)
    # print(gradient(mod_x,mod_theta,y1))


    #computing the cost function

    #computing the gradient

    # grad_f =




if __name__ == '__main__':
    main()



