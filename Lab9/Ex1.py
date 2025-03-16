import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


'''Implement a decision tree regressor using scikit-learn
Implement a decision tree classifier using scikit-learn
'''

def load_data():
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    X = data.drop(columns=["disease_score", "disease_score_fluct"], axis=1).values
    y = data["disease_score"]
    return X, y

def load_data2():
    data = pd.read_csv("/ML_PRACTICE/Breast_cancer_wisconcsin.csv")
    X = data.drop(columns=["diagnosis", "id", "Unnamed: 32"]).values
    data["diagnosis_binary"] = data["diagnosis"].map({'M': 1, 'B': 0}) #transforms diagnosis column into a binary format
    y = data["diagnosis_binary"].values
    return X, y


def train_regression_tree_cv(X_train,X_test, y_train,y_test):
    model = DecisionTreeRegressor()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    mse_scores = cross_val_score(model, X, y, cv=kf)
    print(f'Mean Squared Error scores across folds: {mse_scores}')
    print(f'Mean MSE: {np.mean(mse_scores)}')

    model.fit(X_train, y_train)



    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, filled=True, feature_names=[f"Feature {i}" for i in range(X.shape[1])])
    plt.title("Decision Tree Regressor")
    plt.show()


def train_classification_tree_cv(X_train1, y_train1, X_test1, y_test1):
    model = DecisionTreeClassifier()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    ######instead of the cross_val_score, use for loop and stratified k fold for better control and return the best model



    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f'Accuracy scores across folds: {accuracy_scores}')
    print(f'Mean accuracy: {np.mean(accuracy_scores)}')

    model.fit(X, y)

    ######test-set logic comes here

    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, filled=True, feature_names=[f"Feature {i}" for i in range(X.shape[1])])
    plt.show()

def partition_dataset_based_on_BP():
    thresholds = [80, 78, 82]
    data = pd.read_csv("../simulated_data_multiple_linear_regression_for_ML.csv")
    for t in thresholds:
        upper_partition = data[data["BP"] > t]
        lower_partition = data[data["BP"] <= t]
        upper_partition.to_csv(f"data_filtered_upper{t}.csv", index=False)
        lower_partition.to_csv(f"data_filtered_lower{t}.csv", index=False)
        print(f"Partitioning done for t = {t}:")
        print(f"Lower partition (BP <= {t}): {lower_partition.shape[0]} rows")
        print(f"Upper partition (BP > {t}): {upper_partition.shape[0]} rows\n")

def main():
    # Exercise 1 - partitioning the dataset
    partition_dataset_based_on_BP()

    # Exercise 2 - Regression Decision tree
    X,y=load_data()
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.30, random_state=999)
    print("Mean squared error for Decision tree regressor:")
    train_regression_tree_cv(X,y,X_Train,X_Test)


    # Exercise 3 - Classification Decision Tree
    X1,y1=load_data2()
    X_Train1, X_Test1, y_Train1, y_Test1 = train_test_split(X1, y1, test_size=0.30, random_state=999)
    print("\nAccuracy value for Decision tree classification:")
    train_classification_tree_cv(X_Train1,X_Test1,y_Train1, y_Test1 )


if __name__=="__main__":
    main()