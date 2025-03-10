import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def lo_data():

    df = pd.read_csv("/home/ibab/PycharmProjects/ML-Lab/Breast_cancer_wisconcsin.csv")

    #Extract the features and target
    x = df.drop(columns=["id","diagnosis","Unnamed: 32"])
    y= df["diagnosis"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    return x,y

def main():
    #load the data
    [x,y]= lo_data()

    #split the data

    x_train, x_test , y_train, y_test = train_test_split(x,y,test_size= 0.30,random_state=999)

    #standardize the data

    # Scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)  # transform is a dynamic variable, not yet instantiated therefore yellow
    x_test_scaled = scaler.transform(x_test)

    # train a model

    print(".....Training....")
    model = LogisticRegression()

    # train the model
    model.fit(x_train_scaled, y_train)

    # prediction on a test set
    y_pred = model.predict(x_test_scaled)


    #get coefficients
    normal_theta = model.coef_
    # print(normal_theta.shape)
    print(f"coefficients, {normal_theta}")

    # compute the r2 score

    accuracy = accuracy_score(y_test,y_pred)
    print(f"accuracy: {accuracy*100:.2f}%")

if __name__ == '__main__':
    main()