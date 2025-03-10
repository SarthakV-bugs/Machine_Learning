import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def lo_data():
    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/Breast_cancer_wisconcsin.csv")
    x = df.drop(columns=["id", "diagnosis", "Unnamed: 32"])
    y = df["diagnosis"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return x, y

def l2_norm(theta, lambda_value):
    return (np.sum(theta ** 2))*lambda_value

def l1_norm(theta, lambda_value):
    return np.sum(np.abs(theta))*lambda_value

def main():
    # Load the data
    x, y = lo_data()

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=999)


    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    lambda_value = 0.1

    # Train a model with L2 regularization
    print(".....Training....")
    # model = LogisticRegression(penalty='l1', C=1/lambda_value, solver='liblinear')
    model = LogisticRegression(penalty='l2', C=1 / lambda_value )

    model.fit(x_train_scaled, y_train)


    y_pred = model.predict(x_test_scaled)

    # Get coefficients
    normal_theta = model.coef_[0]  # Get the coefficients for the first class
    print(f"Coefficients: {normal_theta}")

    # Compute L1 and L2
    l1 = l1_norm(normal_theta, lambda_value)
    l2 = l2_norm(normal_theta, lambda_value)
    print(f"L1 norm with lambda: {l1}")
    print(f"L2 norm with lambda: {l2}")


    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()