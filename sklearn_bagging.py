# Implement bagging regressor and classifier using scikit-learn. Use diabetes and iris datasets.
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score

from Lab4.Lab4_Ex3_admission_data import r2_score

##### Bagging for Regression (Diabetes dataset) #####
diabetes = load_diabetes()
X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_diabetes = diabetes.target  # Continuous target

# Splitting data
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)

# Base learner - Decision Tree Regressor
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train_d, y_train_d)
y_pred_tree = tree_regressor.predict(X_test_d)

# Bagging Regressor
bagging_regressor = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=10,
    random_state=42,
    bootstrap=True
)
bagging_regressor.fit(X_train_d, y_train_d)
y_pred_bagging = bagging_regressor.predict(X_test_d)

# Compute Errors
r2_tree = r2_score(y_test_d, y_pred_tree)
r2_bagging = r2_score(y_test_d, y_pred_bagging)

mse_tree = mean_squared_error(y_test_d, y_pred_tree)
mse_bagging = mean_squared_error(y_test_d, y_pred_bagging)

print(f"Decision Tree regressor r2 score: {r2_tree:.2f}")
print(f"Bagging Tree r2 score: {r2_bagging:.2f}")
print(f"\n")
print(f"Decision Tree Regressor MSE: {mse_tree:.2f}")
print(f"Bagging Regressor MSE: {mse_bagging:.2f}")
print(f"\n")





                                                 #####  Bagging for Classification (Iris dataset)   #####
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # Categorical target

# Splitting data
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# Base learner - Decision Tree Classifier
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train_i, y_train_i)
y_pred_tree_i = tree_classifier.predict(X_test_i)

# Bagging Classifier
bagging_classifier = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42,
    bootstrap=True
)
bagging_classifier.fit(X_train_i, y_train_i)
y_pred_bagging_i = bagging_classifier.predict(X_test_i)

# Compute Accuracies
acc_tree = accuracy_score(y_test_i, y_pred_tree_i)
acc_bagging = accuracy_score(y_test_i, y_pred_bagging_i)

print(f"Decision Tree Classifier Accuracy: {acc_tree:.2f}")
print(f"Bagging Classifier Accuracy: {acc_bagging:.2f}")

