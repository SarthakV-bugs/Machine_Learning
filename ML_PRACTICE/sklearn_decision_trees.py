# 1. Write a program to partition a dataset (simulated data for regression) into two parts,
# based on a feature (BP) and for a threshold, t = 80. Generate additional two partitioned
# datasets based on different threshold values of t = [78, 82].
# 2. Implement a regression decision tree algorithm using scikit-learn for the simulated
# dataset.
# 3. Implement a classification decision tree algorithm using scikit-learn for the simulated
# dataset.


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import seaborn as sns


                                                                            ###Question2
''' DECISION TREE REGRESSION MODEL FOR SIMULATED DATASET'''
#Step1: load the dataset

simulated_data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
print(simulated_data)
# print(simulated_data.keys()) #Index(['age', 'BMI', 'BP', 'blood_sugar', 'Gender', 'disease_score','disease_score_fluct'],dtype='object')

#Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.

#Step2: Loading features and target values
'''Extracting the features and target values as x and y by dropping "disease_score" and "disease_score_fluct"'''

# x = simulated_data.drop(columns=['disease_score','disease_score_fluct','BMI','blood_sugar','Gender'])
x = simulated_data.drop(columns=['disease_score','disease_score_fluct'])
# print(x.keys()) #Index(['age', 'BMI', 'BP', 'blood_sugar', 'Gender'], dtype='object')

y1 = simulated_data[['disease_score']]
y2 = simulated_data[['disease_score_fluct']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# print(y1.keys()) #Index(['disease_score'], dtype='object')
# print(y2.keys()) #Index(['disease_score_fluct'], dtype='object')

#Step3: Split the data into train and test set

#data split using x and y2
x_train , x_test, y_train, y_test = train_test_split(x_scaled,y2,test_size = 0.3,random_state = 42)
# print(y_test.shape)
y_test = y_test.to_numpy().ravel() #makes sure the dimensions are same as ypred(n,), y_test is a dataframe of (n,1) dimensions
# print(y_test.shape)

#Step4: Train the model using Decision tree regressor
###model with pruning
# model = DecisionTreeRegressor(max_depth=4, min_samples_split=5, min_samples_leaf=3, random_state=42)
model = DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)


###model without pruning
# model = DecisionTreeRegressor()

model.fit(x_train, y_train)

#Step5: Predict the y_test using the x_test
##prediciton on the training data without pruning gives 100 percent r2score which suggests overfitting,
##on pruning the model, the r2 score for the training itself drops to 78, and the r2score for test increases

ypred1 = model.predict(x_train)

ypred = model.predict(x_test)
# print(ypred.shape)
print(f"ypred:{ypred}")


#Step6: Compute r2 score or MSE
r2score = r2_score(y_test, ypred)
r2score1 = r2_score(y_train.to_numpy().ravel(), ypred1)
print(f"r2score for test data: {r2score}")
print(f"r2score1 for training data: {r2score1}")

# Cross-validation
cv_scores = cross_val_score(model, x_scaled, y2, cv=10, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R² Score: {np.mean(cv_scores):.4f}")

# importance = model.feature_importances_
# plt.bar(x_train.columns, importance)

# plt.xticks(rotation=45)
# plt.show()

                                                           ###Question3####

## Implement a classification decision tree algorithm using scikit-learn for the breast cancer wisconsin dataset.

#Performing some EDA
#loading the dataset
data_cancer = pd.read_csv("Breast_cancer_wisconcsin.csv")
print(data_cancer.keys())

#Know the data

print(data_cancer.info())
print(data_cancer.head(5))
print(data_cancer.tail(5))
print(data_cancer.describe())#gives stats about the dataset
print(data_cancer.isnull().sum()) #for each column it returns the count of missing values
print(data_cancer['diagnosis'].value_counts()) #returns the count of classes

#we can conclude that column , 'id' and 'Unnamed: 32' can be dropped.

#plotting pair plot and heat maps
# data_cancer2 = data_cancer.drop(columns=["diagnosis","id","Unnamed: 32"])
# corr_matrix_breast = data_cancer2.corr(method="pearson") #corr matrix of pandas
# print(corr_matrix_breast)
#
# sns.heatmap(data_cancer2.corr(),cmap = "coolwarm", linecolor = 'white', linewidths = 1,annot=True)
# plt.show()

'''DECISION TREE CLASSIFICATION MODEL FOR BREAST CANCER'''


# Load Cancer Dataset
data_cancer = pd.read_csv("Breast_cancer_wisconcsin.csv")

# Drop Unnecessary Columns
x1 = data_cancer.drop(columns=["Unnamed: 32","id", "diagnosis"])
y = data_cancer['diagnosis']

# Label Encoding (Convert 'M' and 'B' to 0/1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
model_clf = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
model_clf.fit(x1_train, y_train)

# Predictions
ypred_cancer = model_clf.predict(x1_test)

# Evaluation
accuracy = accuracy_score(y_test, ypred_cancer)
print(f"Classification Accuracy: {accuracy:.4f}")
print(classification_report(y_test, ypred_cancer, target_names=label_encoder.classes_))


                                                                                ###Question1
#partition logic to split the datasets on the basis of feature (BP) and threshold value, t=80.

def partition_logic(feature_name, threshold):
    #t stands for the threshold value for the split
    left_partition = [] #list for storing left partition values of the dataset
    right_partition = []  #list for storing right partition values of the dataset

    for i in simulated_data["BP"]: #iterates through the feature BP for partitioning
        if i >= threshold:
            right_partition.append(i)

        else:
            left_partition.append(i)


    print(f"Right partition:{right_partition}")
    print(f"Left partition: {left_partition}")




def main():

##Partitioning the dataset logic
    threshold = [78,80,82]
    for t in threshold:
        print(f"\n Partition on the basis of {t}")
        partition_logic("BP",t)


if __name__ == '__main__':
    main()