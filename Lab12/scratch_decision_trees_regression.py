# Implement a decision regression tree algorithm without using scikit-learn using the
#diabetes dataset. Fetch the dataset from scikit-learn library


import numpy as np
import pandas as pd
from math import inf

from scipy.constants import value
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
###DECISION TREE CLASSIFIER

from sklearn.model_selection import train_test_split

# from Lab10.Ex1 import information_gain


#create a class node.

#class node is constructed as it's used recursively for creating various nodes in the build tree logic which takes on
#different values
#class node should have parameters such as feature_name, threshold value, child nodes, predicted class if it's a node

class Node:
    def __init__(self,feature=None, threshold=None, left_node=None,right_node=None, value=None):
        self.feature = feature              #takes feature index to split on
        self.threshold = threshold          #threshold value
        self.left_node = left_node          #for internal node while splitting, if the value of feature is less than threshold
        self.right_node = right_node
        self.value = value #only for leaf node(class labels)


#LOADING THE DATASET
diabetes = load_diabetes()
#extracting the design matrix as x
x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#extracting target variable as y
y = pd.Series(diabetes.target)

#code block for computing mse
def mse(y):

    mse = np.mean((y - np.mean(y)) ** 2)
    return mse

def decrease_mse(y, left_y, right_y):
    weighted_mse_l = (len(left_y)/len(y)) * mse(left_y)
    weighted_mse_r = (len(left_y)/len(y)) * mse(right_y)
    return mse(y) - (weighted_mse_l + weighted_mse_r)




#code block for computing decrease in mse values
#create a list of midpoints of all the unique values for each feature and store it as a dict.
def compute_thresholds(x):
    thresholds = {} #dicitionary to store each features' unique midpoint values list

    for features in x.columns: #iterate over each column name which represents the features of design matrix x
      unique_values = sorted(x[features].unique()) #each unique value of the feature is sorted
      midpoints = [(unique_values[i] + unique_values[i+1])/2.0 for i in range(len(unique_values) - 1)] #midpoint for each unique value is computed.
      thresholds[features] = midpoints #stores midpoints against the feature name in a dictionary format
    return thresholds



# Code for the split_logic which takes parameters: features and thresholds.
# split logic iterates through all the features and all the possible threshold values i.e. the midpoints of the unique values
# of each feature.
#Intergrates the splitting of y values for each split

def split_logic(x,y, thresholds):

    #initialize best_feature, best_information_gain, best_threshold
    best_feature = None
    best_threshold = None
    best_mse_reduction = -inf

    #splitting logic code goes here
    for feature in x.columns: #iterate over each feature
        for threshold in thresholds[feature]: #iterate over the mid-values of each feature
            #split the dataset on the basis of thresholds
            left_split = x[feature] <= threshold #boolean masks not the actual values
            right_split = x[feature] > threshold
            left_y, right_y = y[left_split], y[right_split]
            # print(left_y)
            ###labels of y based on the split
            #y split for computing information gain to decide the best split

            # Edge case: If any split is empty, continue (skip bad splits)
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            #Calculating the metric such as Information gain to decide the best split
            mse_decrease = decrease_mse(y,left_y,right_y) #takes the labels as parameters calculates the parent entropy and child entropy to give ig
            print(f"feature:{feature},Threshold:{threshold},mse reduction:{mse_decrease}\n")


            #choosing the best split
            if mse_decrease > best_mse_reduction:
                best_feature = feature
                best_threshold = threshold
                best_mse_reduction = mse_decrease
    print(f"Best split => feature : {best_feature}, Threshold: {best_threshold}, best_mse_reduction:{best_mse_reduction}")
    return best_feature , best_threshold , best_mse_reduction

#Recursively built the Decision tree using the split logic and cls Node
def Build_tree(x,y,thresholds,depth=0, max_depth=None):

    #code block for stopping conditions
    if len(y) < 2: #if datapoints are too less
        return Node(value=np.mean(y))

    if max_depth is not None and depth >= max_depth: #if maximum depth exceeded
        return Node(value=y.mode()[0]) #returns the most common class in node


    #code block for best split
    best_feature, best_threshold, best_information_gain = split_logic(x,y,thresholds)

    if best_information_gain <= 0: #no best useful split is found, sets the leaf node values
        return Node(value=y.mode()[0])


    #code block for splitting the dataset using the best split
    left_split = x[best_feature] <= best_threshold
    right_split = x[best_feature] > best_threshold
    left_x, right_x = x[left_split], x[right_split] #getting the values of x
    left_y, right_y = y[left_split], y[right_split] #getting the class labels of y

    #recursively call build tree function on left and right child nodes
    # print(f"Depth {depth}: Splitting on {best_feature} at {best_threshold}")
    left_subtree = Build_tree(left_x, left_y, thresholds, depth+1, max_depth)
    right_subtree = Build_tree(right_x, right_y, thresholds, depth+1, max_depth)

    #return the node containing its feature, split value, left child and right child


    return Node(feature=best_feature,threshold=best_threshold,left_node=left_subtree,right_node=right_subtree)

def predict_value(Node, x):
     #if the current node is a leaf node, return the predicted class
     if Node.value is not None:
         return Node.value

     #if Node is an internal node, traverse the tree based on feature and threshold
     if x[Node.feature] <= Node.threshold:
         return predict_value(Node.left_node, x)
     else:
         return predict_value(Node.right_node, x)

#work on this code block
# def calculate_accuracy(y_true, y_pred):
#     # Ensure that y_true and y_pred are both lists or numpy arrays
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#
#     # Ensure both have the same length
#     if len(y_true) != len(y_pred):
#         raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")
#
#     # Calculate the number of correct predictions
#     correct_predictions = np.sum(y_true == y_pred)
#
#     # Return accuracy as the ratio of correct predictions to total predictions
#     accuracy = correct_predictions / len(y_true)
#
#     return accuracy

#train-test split model evaluation
def train_test_split_evaluation(x,y,test_size = 0.3, max_depth=None):
    #split the dataset into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=42)


    #Build the DT using the x_train
    thresholds = compute_thresholds(x_train)
    tree = Build_tree(x_train, y_train, thresholds, max_depth=max_depth)

    #make predicitions on the test data
    y_pred = [predict_value(tree,x_test.iloc[i]) for i in range(len(x_test))]


    return mean_squared_error(y_pred,y_test)

def main():
    thresholds = compute_thresholds(x)
    print(f"Thresholds: {thresholds}")
    print(split_logic(x,y, thresholds=compute_thresholds(x)))
    tree = Build_tree(x,y,thresholds=compute_thresholds(x),max_depth=3)
    print(tree)

    # # Make predictions for the first 5 samples in the dataset
    # predictions = [predict_class(tree, x.iloc[i]) for i in range(10)]
    # print("Predictions:", predictions)
    #
    # # Ground truth for comparison
    # print("True labels:", y.iloc[:10].values)

    #train_test_eval_acc

    mse_error = train_test_split_evaluation(x,y,test_size=0.3,max_depth=5)
    print(f"mse: {mse_error}")

if __name__ == '__main__':
    main()
#
# import numpy as np
# import pandas as pd
# from math import inf
# from sklearn.datasets import load_diabetes
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
#
#
# # Node class for the decision tree
# class Node:
#     def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, value=None):
#         self.feature = feature
#         self.threshold = threshold
#         self.left_node = left_node
#         self.right_node = right_node
#         self.value = value  # Only for leaf nodes
#
#
# # Load the diabetes dataset
# diabetes = load_diabetes()
# X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# y = pd.Series(diabetes.target)
#
#
# # Compute Mean Squared Error (MSE)
# def mse(y):
#     return np.mean((y - np.mean(y)) ** 2)
#
#
# # Calculate the reduction in MSE when splitting
# def decrease_mse(y, left_y, right_y):
#     weighted_mse_l = (len(left_y) / len(y)) * mse(left_y)
#     weighted_mse_r = (len(right_y) / len(y)) * mse(right_y)
#     return mse(y) - (weighted_mse_l + weighted_mse_r)
#
#
# # Compute possible threshold values (midpoints of unique feature values)
# def compute_thresholds(X):
#     thresholds = {}
#     for feature in X.columns:
#         unique_values = sorted(X[feature].unique())
#         midpoints = [(unique_values[i] + unique_values[i + 1]) / 2.0 for i in range(len(unique_values) - 1)]
#         thresholds[feature] = midpoints
#     return thresholds
#
#
# # Determine the best split based on MSE reduction
# def split_logic(X, y, thresholds):
#     best_feature, best_threshold, best_mse_reduction = None, None, -inf
#
#     for feature in X.columns:
#         for threshold in thresholds[feature]:
#             left_split = X[feature] <= threshold
#             right_split = X[feature] > threshold
#             left_y, right_y = y[left_split], y[right_split]
#
#             if len(left_y) == 0 or len(right_y) == 0:
#                 continue
#
#             mse_decrease = decrease_mse(y, left_y, right_y)
#
#             if mse_decrease > best_mse_reduction:
#                 best_feature, best_threshold, best_mse_reduction = feature, threshold, mse_decrease
#
#     return best_feature, best_threshold, best_mse_reduction
#
#
# # Recursively build the decision tree
# def build_tree(X, y, thresholds, depth=0, max_depth=None):
#     if len(y) < 2:
#         return Node(value=np.mean(y))
#
#     if max_depth is not None and depth >= max_depth:
#         return Node(value=np.mean(y))
#
#     best_feature, best_threshold, best_mse_reduction = split_logic(X, y, thresholds)
#
#     if best_mse_reduction <= 0:
#         return Node(value=np.mean(y))
#
#     left_split = X[best_feature] <= best_threshold
#     right_split = X[best_feature] > best_threshold
#     left_X, right_X = X[left_split], X[right_split]
#     left_y, right_y = y[left_split], y[right_split]
#
#     left_subtree = build_tree(left_X, left_y, thresholds, depth + 1, max_depth)
#     right_subtree = build_tree(right_X, right_y, thresholds, depth + 1, max_depth)
#
#     return Node(feature=best_feature, threshold=best_threshold, left_node=left_subtree, right_node=right_subtree)
#
#
# # Predict the output value for a given sample
# def predict_value(node, x):
#     if node.value is not None:
#         return node.value
#     if x[node.feature] <= node.threshold:
#         return predict_value(node.left_node, x)
#     else:
#         return predict_value(node.right_node, x)
#
#
# # Train-test split and model evaluation
# def train_test_split_evaluation(X, y, test_size=0.3, max_depth=None):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     thresholds = compute_thresholds(X_train)
#     tree = build_tree(X_train, y_train, thresholds, max_depth=max_depth)
#     y_pred = [predict_value(tree, X_test.iloc[i]) for i in range(len(X_test))]
#     return mean_squared_error(y_test, y_pred)
#
#
# # Main function to run the model
# def main():
#     thresholds = compute_thresholds(X)
#     tree = build_tree(X, y, thresholds, max_depth=3)
#     mse_error = train_test_split_evaluation(X, y, test_size=0.3, max_depth=5)
#     print(f"Mean Squared Error: {mse_error}")
#
#
# if __name__ == '__main__':
#     main()