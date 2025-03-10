# Implement a linear regression model using scikit-learn for the simulated dataset
# - simulated_data_multiple_linear_regression_for_ML.csv  - to predict the disease
# score from multiple clinical parameters

#Perform EDA

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns




#create a panda dataframe
def load_data():
    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")
    print(df)

    X = df.drop(columns=["disease_score","disease_score_fluct"]) #X features
    Y1 = df["disease_score"]
    Y2 = df["disease_score_fluct"]

    return X,Y1,Y2

def eda():
    sample_data = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/simulated_data_multiple_linear_regression_for_ML.csv")
    print(sample_data)

    print(sample_data.head())
    print(sample_data.tail())

    print(sample_data.iloc[:,0:-2])
    print(sample_data.iloc[:, -2:])

    #plot
    sample_data.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()

    print(sample_data.describe())

    sns.scatterplot(
        data=sample_data,
        x="age",
        y="BP",
        size="disease_score",
        hue="disease_score",
        palette="viridis",
        alpha=0.5,
    )
    plt.legend(title="disease_score", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    _ = plt.title(
        "Disease score based on age and bp")  # _ captures the plt.title, for aesthetic purposes
    plt.show()

    print(sample_data.corr())

    #subplot

    pd.plotting.scatter_matrix(sample_data[["age","BMI","BP","blood_sugar","Gender","disease_score","disease_score_fluct"]])
    # pd.plotting.scatter_matrix(sample_data[["age","disease_score","disease_score_fluct"]])
    plt.show()

    #plot bw age and disease score, disease score_fluct

    sns.scatterplot(
        data=sample_data,
        x="age",
        y="disease_score",
        alpha=0.5,
    )
    plt.legend(title="disease_score", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    _ = plt.title(
        "Disease score vs age")  # _ captures the plt.title, for aesthetic purposes
    plt.show()

    sns.scatterplot(
        data=sample_data,
        x="age",
        y="disease_score_fluct",
        alpha=0.5,
    )
    plt.legend(title="disease_score_fluct", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    _ = plt.title(
        "Disease score_fluct vs age")  # _ captures the plt.title, for aesthetic purposes
    plt.show()



def main():
    # [X,Y1,Y2] = load_data()
    [X,Y1,Y2] = load_data()


    #split the data
    x_train = pd.read_csv('x_train.csv', header=None).to_numpy()
    x_test = pd.read_csv('x_test.csv', header=None).to_numpy()
    y_train = pd.read_csv('y_train.csv', header=None).to_numpy()
    y_test = pd.read_csv('y_test.csv', header=None).to_numpy()
    # x_train, x_test, y_train, y_test = train_test_split(X, Y2, test_size=0.30, random_state=999)



    # Scale the data
    # scaler = StandardScaler()
    # scaler = scaler.fit(x_train)
    # x_train_scaled = scaler.transform(x_train)
    # x_test_scaled = scaler.transform(x_test)

    ##train a model

    print(".....Training....")
    model = LinearRegression()

    # train the model
    # model.fit(x_train_scaled, y_train)
    model.fit(x_train, y_train)


    # prediction on a test set
    y_pred = model.predict(x_test)
    # y_pred = model.predict(x_test_scaled)

    #get coefficients
    normal_theta = model.coef_
    print(f"coefficients, {normal_theta}")

    # compute the r2 score

    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f(closer to 1 is good) " % r2)

    eda()
    ''' Train the model with feature having the most confidence '''

if __name__ == '__main__':
    main()