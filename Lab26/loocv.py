# #weekly dataset, perform LOOCV
# import numpy as np
# import pandas as pd
# from ISLP import load_data
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
#
# #load the dataset
# weekly = load_data('Weekly')
# print(weekly.info())
# print(weekly.head(5))
#
# #design matrix x
# x = weekly[['Lag1','Lag2']]
# print(x)
#
# #direction is the target variable
# y = weekly['Direction']
# # print(y)
#
# #label encode the target y
# le = LabelEncoder()
# y = pd.Series(le.fit_transform(y))
# # print(y)
#
#
#
#
# ##a)
# #perform LOOCV
# y_pred = []
# for i in range(len(x)):
#     x_train = x.drop(index=i)
#     y_train = y.drop(index=i)
#     x_test = x.iloc[[i]] #2d
#     y_test = y.iloc[i]
#
#     clf = LogisticRegression()
#     clf.fit(x_train,y_train)
#     ypred = clf.predict(x_test)
#     y_pred.append(ypred[0]) #access the scalar values from the array
#
#
# print(pd.Series(y_pred))
#
#
# ##b)
# #check the accuracy
# y_pred_correct = []
# for i in range(len(y_pred
#                    )):
#     if y_pred[i] == y.iloc[i]:
#         y_pred_correct.append(y_pred[i])
#
# print(y_pred_correct)
#
#
# accuracy = len(y_pred_correct) / len(y_pred)
# print(accuracy)
#
#
#
#
#
#
#
#
#
#
#


import numpy as np
from ISLP import load_data
from sklearn.linear_model import LogisticRegression

# Load data
weekly = load_data('Weekly')
X = weekly[['Lag1', 'Lag2']]
y = weekly['Direction']

# Initialize list to store individual accuracies
individual_accuracies = []

# LOOCV
for i in range(len(X)):
    # Split data (leave-one-out)
    X_train = X.drop(index=i)
    y_train = y.drop(index=i)
    X_test = X.iloc[[i]]
    y_test = y.iloc[i]

    # Fit model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate accuracy for this fold (1 if correct, 0 if wrong)
    pred = model.predict(X_test)
    individual_accuracies.append(1 if pred[0] == y_test else 0)

# Calculate LOOCV estimate as average of individual accuracies
loocv_accuracy = np.mean(individual_accuracies)
print(f"LOOCV estimate for test accuracy: {loocv_accuracy:.4f} ({(loocv_accuracy * 100):.2f}%)")