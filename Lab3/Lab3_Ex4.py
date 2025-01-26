import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from pandas.core.window.doc import template_header


#Use the above simulated CSV file and implement the following from scratch in Python
# Read simulated data csv file
# Form x and y
# Write a function to compute hypothesis
# Write a function to compute the cost
# Write a function to compute the derivative
# Write update parameters logic in the main function


#read the file & form x and y using libraries

def load_data():
    #read the csv file
    df = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    # print(df)

    #initialize the data
    #add first column as x_ones to set the intercept

    x = df.drop(columns=["disease_score", "disease_score_fluct"])

    #create x_ones
    x_ones = []
    for i in range(len(x)): #iterated over the rows
        x_ones.append(1)

    #converted the list into a df and concatenated.
    x_ones_df = pd.DataFrame({'x_ones': x_ones})
    x = pd.concat([x_ones_df,x],axis=1).to_numpy()
    # x = pd.concat([x_ones_df, x], axis=1)
    # print(x)
    # print(type(x))
    # print(len(x))
    # Normalize the data (scaling features to mean 0 and variance 1)
    mean = np.mean(x[:, 1:], axis=0)  # Compute the mean of each feature (excluding the intercept)
    std_dev = np.std(x[:, 1:], axis=0)  # Compute the standard deviation of each feature (excluding the intercept)

    # Normalize each feature (excluding the first column which is the intercept term)
    x[:, 1:] = (x[:, 1:] - mean) / std_dev

    y1 = df["disease_score"].to_numpy()
    y2 = df["disease_score_fluct"].to_numpy()

    return x,y1,y2

def initialize_para(x):

    theta = []
    random.seed(42)
    for i in x[0,:]: #iterated over the cols
        x_theta = random.uniform(a=0,b=0.1)
        theta.append(x_theta)
    return theta


def hypo(x,theta):

    #my code using array manipulation
    h_xes = []
    for i in range(len(x)):
        h_x = 0
        for j in range(len(theta)):
            if j >= len(x[0,:]): #to prevent out of bound error, no of columns.
                break
            h_x += theta[j] * x[i][j]
        h_xes.append(h_x)
    return h_xes

    # h_xes = []
    # for index, row in x.iterrows():  #Iterate through each row in the DataFrame
    #     h_x = 0
    #     for j, feature in enumerate(row):  #Iterate through each feature in the row
    #         h_x += theta[j] * feature
    #     h_xes.append(h_x)
    # return pd.Series(h_xes)

def cost(h_xes,y1,y2):
    sse = 0
    for j in range(len(h_xes)):
        err = (h_xes[j] - y1[j]) #error output
        sse += err**2 #squared error
    return  sse #return sum of squared error


def grad_f(error,x):
    gd = [0] * len(x[0])
    for i in range(len(error)):
        for j in range(len(x[i])):
            gd[j] += error[i] * x[i,j] #each feature of x, we get the added final gradient for each feature
    n = len(error) #total samples
    gd = [grad / n for grad in gd] #average gradient
    return gd


def update_parameters(theta,alpha,gd):

    upd_theta = [] #updated theta value
    for i in range(len(theta)):
        theta_2 = theta[i] - alpha * gd[i]
        upd_theta.append(theta_2)
    return upd_theta

def plotting(cost_history,iterations):
    plt.plot(range(iterations), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Iterations')
    plt.show()


def main():

    alpha = 0.01
    x,y1,y2 = load_data()
    # print(x)
    theta = initialize_para(x)
    # print(len(theta))
    # print(theta)

    # print(hypo(x,theta))
    h_xes = hypo(x,theta)
    hxes = np.array(h_xes)
    # print(h_xes) #np.float64(9.466440254802466), np.float64(9.224307113857158), np.float64(8.890919763256665), np.float64(9.180662279960375), np.float64(9.040942736687038
    # print(y1)
    # print(cost(h_xes,y1,y2))

    #load the error

    error = hxes - y1
    # print(error)
    # error = [hxes - y1 for h_x,y_i  in zip(h_xes,y1)]
    # print(error)

    #calling grad
    gd = grad_f(error,x)
    # print(grad_f(error,x))

    #parameters updation
    upd_theta = update_parameters(theta,alpha, gd)

    cost_history = []
    iterations = 1000
    for i in range(iterations):

        #compute hypothesis
        h_xes = hypo(x, theta)
        hxes = np.array(h_xes)

        #compute cost
        cost_f =  cost(hxes, y1, y2)

        # load the error
        error = hxes - y1

        #compute the gradient
        gd = grad_f(error, x)

        #parameters update
        theta = update_parameters(theta,alpha, gd)
        cost_history.append(cost_f)


        if i % 100 == 0:
            # Print every 100 iterations
            print(f"Iteration {i}: Cost = {cost_f}")

    print("Final theta values:", theta)

    plotting(cost_history,iterations)
    return theta






if __name__ == '__main__':
    main()

