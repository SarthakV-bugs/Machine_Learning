## Implement bagging regressor without using scikit-learn

#write a class for bagging regressor to include all the functionalities needed to implement bagging.
"""requirements to do bagging:

"initialization of the parameters in the class bagging"
 n_estimators -> for the no. of models required
 max_samples ->  for the subset size to be used for making the new dataset
 eg: new_dataset = max_samples * len(x) #length of the original dataset, max_samples is the ratio
 models -> a list which will contain all the learned decision model
 max_depth -> depth allowed for building each base learner

 "training each model on the various bags created using bootstrap sampling with replacement i.e. we fit the
 data set"

 requirements: logic to do bootstrap sampling,iterate through each bag, predict the score using each base model
 and append to the models list.

"prediction based on the average of the scores in model list needs to be reported.
 """

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import  load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score

#load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class Bagging:
    ##setting ne_estimators to a default value
    def __init__(self, n_estimators=10,models=None):
        self.n_estimators = n_estimators #the number of base learners to be trained
        # self.max_samples = max_samples
        self.models = [] #list to store the trained base learner
        # self.max_depth = max_depth

    def bootstrapped(self,x,y):
        """code block to sample the data
        returns bags of all the sampled datasets"""
        sample_size = len(x)  # keeping the sample size same as the dataset
        bags = [] #creating a list of bag to store the indices of the multiple bootstrapped dataset

        for i in range(self.n_estimators):
            #randomly select the dataset of size = sample_size
            sample_indices = np.random.choice(len(x),size=sample_size,replace=True)
            x_sample = x[sample_indices] #x_sampled
            y_sample = y[sample_indices] #y_sampled
            bags.append([x_sample,y_sample])
        return bags


    def fit(self,x,y,bootstrapped):
        """training the weak learners on the bootstrapped sampled datasets"""
        bt_bags = bootstrapped(x,y) #calling

        self.models = []
        for i, b in enumerate(bt_bags):

                base_learner = DecisionTreeRegressor()
                base_learner.fit(b[0], b[1])
                self.models.append(base_learner)

                # Compute R² score for this base model on the test set
                y_pred_base = base_learner.predict(X_test)
                r2_score_base = r2_score(y_test, y_pred_base)
                print(f"R² score for bootstrapped dataset {i + 1}: {r2_score_base}")


    def predict(self,x):
        """predicting the score for each model trained on the bootstrapped dataset and reporting average"""

        prediction = [] #list to store the prediction of each base learner
        for model in self.models:
            pred = model.predict(x)
            prediction.append(pred)


        prediction = np.array(prediction)   #converted to array
        average_prediction = np.mean(prediction,axis=0)    #axis=0 calculates the mean for each sample across all models

        return average_prediction



# Create an instance of the Bagging class
# bagging_model = Bagging(n_estimators=10)
bagging_model = Bagging(n_estimators=200)

# Define the bootstrapped function
bootstrapped = bagging_model.bootstrapped

# Fit the model on the training data
bagging_model.fit(X_train, y_train, bootstrapped)


# Predict the labels for the test set
y_pred = bagging_model.predict(X_test)

"""Calculating r2score for a single base learner"""
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred_bl = model.predict(X_test)

r2_base_learner = r2_score(y_test,y_pred_bl)

print(f"R2 score for single decision tree: {r2_base_learner}")


# Evaluate the model's accuracy
r2 = r2_score(y_test, y_pred)
print(f"r2score of the bagging average: {r2}")

