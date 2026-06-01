# 1. Implement Adaboost classifier using scikit-learn. Use the Iris dataset.
from pandas.core.ops.missing import mask_zero_div_zero
from sklearn.datasets import load_iris
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

#load iris dataset

iris = load_iris()
x = iris.data
y = iris.target

#split the dataset into train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#train Adaboost model
#by default adaboostclassifier takes Decisiontreeclassifier with max_depth as 1.
adaboost_clf = AdaBoostClassifier(n_estimators=50,random_state=42)
#trying different base model
# base_clf = DecisionTreeClassifier(max_depth=2)
# adaboost_clf = AdaBoostClassifier(estimator=base_clf,n_estimators=50,random_state=42)
adaboost_clf.fit(x_train,y_train)


#predict on test set

y_pred = adaboost_clf.predict(x_test)

# Create DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(results_df)


#Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")


#Confusion matrix and Classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#feature importance
feature_importances = adaboost_clf.feature_importances_
plt.bar(iris.feature_names, feature_importances)
plt.xticks(rotation=45)
plt.title("Feature Importances")
plt.show()

