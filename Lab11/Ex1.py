# # Implement decision tree classifier without using scikit-learn using the iris dataset. Fetch
# # the iris dataset from scikit-learn library.
# from ast import increment_lineno
# from pyexpat import features
#
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap
# # from scipy.conftest import pytest_runtest_setup
# from sklearn.datasets import load_iris
# import seaborn as sns
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, plot_tree
#
# iris_data = load_iris()
#
# print(type(iris_data))
# print(iris_data.keys())
# print(iris_data['feature_names'])
# print(iris_data['target_names'])
#
# iris_df = pd.DataFrame(data = iris_data['data'], columns = iris_data['feature_names'])
# print(iris_df) #prints the entire dataframe
# print(iris_df.shape) #dimensions of the data
# print(iris_df.head(5)) #top 5 samples
# print(iris_df.sample(10))  #gives random 10 samples
# print(iris_df.columns) #names of the columns in a list format
#
# #slicing the dataset
# print(iris_df[10:20]) #10th sample will be inclusive and 20th won't be included.
#
# #storing the sliced data into a variable.
# sliced_data = iris_df[10:20]
# print(sliced_data.shape)
#
# #slicing specific columns
# #using the column names
# specific_data = iris_df[["sepal length (cm)", "petal width (cm)"]]
# print(specific_data.head(10))
#
# #using iloc and loc for filtering options
#
# #iris_df["sepal length (cm)"]==0.3 returns a boolean series
# #which in turn is checked by .loc[], if true, it prints the specific row.
#
# print(iris_df.loc[iris_df["sepal length (cm)"]==0.3])
#
# print(iris_df.iloc[1])
#
# #value.counts()
#
# print(iris_df["sepal length (cm)"].value_counts())
#
# #mean, median and sum
# print(iris_df["sepal length (cm)"].sum())
# print(iris_df["sepal length (cm)"].mean())
# print(iris_df["sepal length (cm)"].median())
#
#
# #adding a new column
#
# cols = iris_df.columns
# print(cols)
#
# cols = cols[1:3] #selected the columns from index 0 to 3
# print(iris_df[cols])
#
# #created a new dataframe for the extracted cols
# new_iris_df  = iris_df[cols]
#
# #adding a new column in the dataframe
#
# iris_df["Total of col 1 and 3"] = new_iris_df[cols].sum(axis=1)
# print(iris_df)
#
#
# #detecting missing values
#
# print(iris_df.isnull().head())
#
#
# #generating a heatmap using seaborn library
# iris = sns.load_dataset("iris")
# #
# # iris_numeric = iris.select_dtypes(include=["number"])
# # sns.heatmap(iris_numeric.corr(),cmap = "YlGnBu", linecolor = 'white', linewidths = 1,annot=True)
# # plt.show()
# #
# # #using pandas corr
# #
# # corr_matrix = iris_df.corr(method="pearson")
# # print(corr_matrix)
# #
# # #Pair plot is used to visualize the relationship between each type of column variable
# # g = sns.pairplot(iris,hue="species")
# # print(iris.keys())
# # plt.show()
#
# ''' On analysis of the pair plot, we can make an assertion that petal features such as petal length and petal width have better
# distinguishing effect on the species, using only petal features for classification model will give higher accuracy. '''
#
# ####BUILDING A LOGISTIC CLASSIFIER FOR IRIS DATASET USING ONLY PETAL FEATURES.
#
#
# #STEP 1: FEATURE EXTRACTION
#
# #extracting the petal features as x
# x = iris[["petal_width","petal_length","sepal_width","sepal_length"]]
# print(x)
# #extracting the target column as y
# y = iris["species"]
# # print(y)
#
# #STEP 2: ENCODING TARGET VARIABLE (STRING INTO NUMBER)
#
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)
# print(y)
#
# #for checking which label has been assigned which value
# #loops through all the labels and encoded value pair
# for label, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
#     print(f"{label} -> {encoded_value}")
#
# #alternatively, we can use just
# print(label_encoder.classes_)
#
# #STEP 3: SPLIT DATA INTO TRAIN AND TEST
#
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
#
# #STEP 4: TRAIN A LOGISTIC REGRESSION MODEL
# model = LogisticRegression()
# model.fit(x_train,y_train)
#
# #training a decision tree model
# model2 = DecisionTreeClassifier(max_depth=4,min_samples_split=5,min_samples_leaf=2, random_state=42)
#
#
# ###cross validation
# # scores = cross_val_score(model, x, y, cv=10)
# scores = cross_val_score(model, x_train, y_train, cv=10)
# # scores2 = cross_val_score(model2, x, y, cv=10)
# scores2 = cross_val_score(model2, x_train, y_train, cv=10)
# # Print scores for each fold
# for i, score in enumerate(scores, 1):
#     print(f"Fold {i}: Accuracy = {score:.4f}")
#
# # Print mean and standard deviation
# print(f"\nMean Accuracy: {np.mean(scores):.4f}")
# print(f"Standard Deviation: {np.std(scores):.4f}")
#
# for i, score in enumerate(scores2, 1):
#     print(f"Fold {i}: Accuracy = {score:.4f}")
#
# # Print mean and standard deviation
# print(f"\nMean Accuracy: {np.mean(scores2):.4f}")
# print(f"Standard Deviation: {np.std(scores2):.4f}")
#
# # print(f"Cross-validation accuracy: {scores.mean():.2f}")
# model2.fit(x_train,y_train)
#
#
# ##STEP 5: MAKE PREDICTIONS
# #predicts the target value by using x_test
# y_pred = model.predict(x_test)
# y_pred2 = model2.predict(x_test)
#
# #STEP 6: EVALUATE MODEL PERFORMANCE
#
# accuracy = accuracy_score(y_test,y_pred)
# accuracy2 = accuracy_score(y_test,y_pred2)
# print(accuracy*100)
# print(accuracy2*100)
# print("\nClassification report: \n",classification_report(y_test, y_pred, target_names=label_encoder.classes_))
#
# ###plot tree for decision tree classifier
# plt.figure(figsize=(12, 8))
# plot_tree(model2, feature_names=iris_data.feature_names, class_names=iris_data.target_names, filled=True)
# plt.title("Decision Tree for Iris Dataset")
# plt.show()
#
# # def plot_decision_boundary(x, y, model):
# #     x_min, x_max = x.iloc[:, 0].min() - 0.5, x.iloc[:, 0].max() + 0.5
# #     y_min, y_max = x.iloc[:, 1].min() - 0.5, x.iloc[:, 1].max() + 0.5
# #     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
# #                          np.linspace(y_min, y_max, 200))
# #
# #     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# #     Z = Z.reshape(xx.shape)
# #
# #     plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['blue', 'orange', 'green']))
# #     scatter = plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap=ListedColormap(['blue', 'orange', 'green']), edgecolors='k')
# #
# #     plt.xlabel("Petal Length")
# #     plt.ylabel("Petal Width")
# #     plt.title("Decision Boundaries for Iris Classification")
# #     plt.legend(handles=scatter.legend_elements()[0], labels=list(label_encoder.classes_))
# #     plt.show()
# #
# # plot_decision_boundary(x, y, model)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load the iris dataset
iris_data = load_iris()

# Convert to DataFrame
iris_df = pd.DataFrame(data=iris_data['data'], columns=iris_data['feature_names'])

# Display basic info
print("Dataset shape:", iris_df.shape)
print("Column names:", iris_df.columns.tolist())
print("Target classes:", iris_data['target_names'])
print(iris_df.head())

# Encode target labels
y = iris_data['target']  # No need to manually encode as the dataset is already numeric
x = iris_df  # Features remain the same

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(x_train, y_train)

# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(max_depth=4, min_samples_split=5, min_samples_leaf=2, random_state=42)
decision_tree_model.fit(x_train, y_train)

# Perform cross-validation on training data
log_reg_scores = cross_val_score(log_reg_model, x_train, y_train, cv=10)
dt_scores = cross_val_score(decision_tree_model, x_train, y_train, cv=10)

# Display cross-validation results
def print_cv_results(model_name, scores):
    print(f"\n{model_name} Cross-Validation Results:")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: Accuracy = {score:.4f}")
    print(f"Mean Accuracy: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}")

print_cv_results("Logistic Regression", log_reg_scores)
print_cv_results("Decision Tree", dt_scores)

# Make predictions
y_pred_log_reg = log_reg_model.predict(x_test)
y_pred_dt = decision_tree_model.predict(x_test)

# Evaluate models
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print(f"\nLogistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%")
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")

print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_log_reg, target_names=iris_data['target_names']))
print("\nClassification Report for Decision Tree:\n", classification_report(y_test, y_pred_dt, target_names=iris_data['target_names']))

# Plot decision tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_model, feature_names=iris_data.feature_names, class_names=iris_data.target_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()
