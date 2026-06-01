# Implement Random Forest algorithm for regression and classification using scikit-learn.
# Use diabetes and iris datasets.
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

#load the iris dataset for classification
iris = load_iris()
x_clf = iris.data
y_clf = iris.target

#load the diabetes for regression
diabetes = load_diabetes()
x_reg = diabetes.data
y_reg = diabetes.target

#splitting the datasets into train and test samples

x_train_clf,x_test_clf,y_train_clf,y_test_clf = train_test_split(x_clf,y_clf,test_size=0.2,random_state=42)

x_train_reg,x_test_reg,y_train_reg,y_test_reg = train_test_split(x_reg,y_reg,test_size=0.2,random_state=42)

#Random forest

#Rf regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,random_state=42
)
rf_regressor.fit(x_train_reg,y_train_reg)

#prediction
y_pred_reg=rf_regressor.predict(x_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Regression - Diabetes Dataset")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}\n")


#Rf classifier

rf_classifier = RandomForestClassifier(
    n_estimators=100,random_state=42
)
rf_classifier.fit(x_train_clf,y_train_clf)

#prediction
y_pred_clf = rf_classifier.predict(x_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)

print(f"Classification - Iris Dataset")
print(f"Accuracy: {accuracy:.4f}")

