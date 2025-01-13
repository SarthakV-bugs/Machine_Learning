import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
# from



def load_data():
    [X,y]= fetch_california_housing(return_X_y=True)
    return X, y



def main():
    #load california housing datasets
    [X, y] = load_data()

    #split data
    x_train, x_test , y_train, y_test = train_test_split(X,y,test_size= 0.30,random_state=999)

    # Scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train) #transform is a dynamic variable, not yet instantiated therefore yellow
    x_test_scaled = scaler.transform(x_test)


    #train a model

    print(".....Training....")
    model = LinearRegression()

    #train the model
    model.fit(x_train, y_train)

    #prediction on a test set
    y_pred = model.predict(x_test)

    #compute the r2 score

    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f(closer to 1 is good) " % r2)

if __name__ == '__main__':
    main()