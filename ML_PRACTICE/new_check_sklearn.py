import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
iris = pd.read_csv("Iris.csv")

# Extract features and target
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris["Species"]

# Encode target variable
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#one vs rset classifier


# Train a Logistic Regression model
log_model = OneVsOneClassifier(LogisticRegression(max_iter=1000))
log_model3 = OneVsRestClassifier(LogisticRegression(max_iter=1000))
# OneVsOneClassifier(log_model)
log_model.fit(X_train, y_train)
log_model3.fit(X_train,y_train)
# Train a Decision Tree model
# dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2, random_state=42)
# dt_model.fit(X_train, y_train)

# Perform cross-validation
log_cv_scores = cross_val_score(log_model, X_train, y_train, cv=10)
# dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=10)

# Print cross-validation results
def print_cv_results(model_name, scores):
    print(f"{model_name} Cross-Validation Results:")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: Accuracy = {score:.4f}")
    print(f"\nMean Accuracy: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}\n")

print_cv_results("Logistic Regression", log_cv_scores)
# print_cv_results("Decision Tree", dt_cv_scores)

# Make predictions
y_pred_log = log_model.predict(X_test)
# y_pred_dt = dt_model.predict(X_test)
y_pred_log2 = log_model3.predict(X_test)
# Evaluate model performance
accuracy_log = accuracy_score(y_test, y_pred_log)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_log2 = accuracy_score(y_test,y_pred_log2)
print(f"Logistic Regression Accuracy onevsone: {accuracy_log * 100:.2f}%")
print(f"Logistic Regression Accuracy onevsrest: {accuracy_log2 * 100:.2f}%")
# print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log, target_names=y_encoder.classes_,output_dict=False))

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# def evaluate(model, x_train, x_test, y_train, y_test):
#     y_test_pred = model.predict(x_test)
#     y_train_pred = model.predict(x_train)
#
#     print("TRAINING RESULTS: \n===============================")
#     clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
#     print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
#     print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
#     print(f"CLASSIFICATION REPORT:\n{clf_report}")
#
#     print("TESTING RESULTS: \n===============================")
#     clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
#     print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
#     print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
#     print(f"CLASSIFICATION REPORT:\n{clf_report}")
#
# def load_data(df_path):
#     df = pd.read_csv(df_path)
#     df['diagnosis'] = df['diagnosis'].map({"B": 0, "M": 1})
#
#     x = df.drop(columns=['diagnosis'], axis=1)
#     y = df['diagnosis']
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#
#     return x_train, x_test, y_train, y_test
#
# def bagging(x_train, x_test, y_train, y_test):
#     tree = DecisionTreeClassifier()
#     bagging_clf = BaggingClassifier(estimator=tree, n_estimators=100, random_state=42)
#     bagging_clf.fit(x_train, y_train)
#
#     evaluate(bagging_clf, x_train, x_test, y_train, y_test)
#
#     scores = {
#         'Bagging Classifier': {
#             'Train': accuracy_score(y_train, bagging_clf.predict(x_train)),
#             'Test': accuracy_score(y_test, bagging_clf.predict(x_test)),
#         },
#     }
#
#     print("SCORES:")
#     print(scores)
#
# def main():
#     # Load data
#     df_path = "/home/ibab/datasets_nope/data.csv"
#     x_train, x_test, y_train, y_test = load_data(df_path)
#     bagging(x_train, x_test, y_train, y_test)
#
# if __name__ == "__main__":
#     main()