# #K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
# Data normalization - scale the values between 0 and 1. Implement code from scratch.
# Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
# Use validation set to do feature and model selection.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

#use sonar dataset to do 10 fold cross validation i.e. k = 10
#target variables are rock or metal designated as R and M

#load the dataset
data = pd.read_csv("../ML_PRACTICE/sonar.csv")
# print(data.info())
print(data.keys())
# print(data.head(5))

#extract X and y
x = data.drop(columns=['R'])
y = data['R']
# print(x.head(5))
# print(y.head(5))

#standardize the dataset
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# print(x_scaled)

#
#now perform kfold cv using sklearn
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
#
#train the model
clf = LogisticRegression(max_iter=1000)

#use the model to perform cross validation
scores = cross_val_score(clf, x_scaled,y,cv=kf, scoring='accuracy')

#calculate the average accuracy for each fold
avg_acc = np.average(scores)

print(round(avg_acc,4))
print(f"Scores for each fold:", [round(score, 4)for score in scores])



#perform hyperparameter tuning, model selection on the above dataset using a validation set of 20 percent

#logisitc regression
lr = LogisticRegression(max_iter=1000)
c_values = np.random.uniform(0.1,10,10)
lr_params = {
    'C': c_values,
    'penalty' : ['l1','l2'],
    'solver': ['liblinear']
}

#perform hyperparameter tuning on the model using GridSearchCV

grid = GridSearchCV(lr,lr_params,scoring='accuracy',cv=5)
grid.fit(x_scaled,y)

print(grid.best_params_)
print(grid.best_score_)

#using the best parameters obtained, test on the test sample

lr = LogisticRegression(C='4.077',
                        penalty='l2',
                        solver='liblinear')
lr.fit(x_scaled,y)
y_pred = lr.predict(x_scaled)
print(accuracy_score(y, y_pred))