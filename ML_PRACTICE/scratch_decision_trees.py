# 1. Implement decision tree classifier without using scikit-learn using the iris dataset. Fetch
# the iris dataset from scikit-learn library.
#2. Implement a decision regression tree algorithm without using scikit-learn using the
#diabetes dataset. Fetch the dataset from scikit-learn library.
import numpy as np
import pandas as pd
from numpy.ma.extras import unique
from math import inf
###DECISION TREE CLASSIFIER

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from Lab10.Ex1 import information_gain
from ML_PRACTICE.sklearn_decision_trees import x_train, accuracy


#create a class node.

#class node is constructed as it's used recursively for creating various nodes in the build tree logic which takes on
#different values
#class node should have parameters such as feature_name, threshold value, child nodes, predicted class if it's a node

class Node:
    def __init__(self,feature=None, threshold=None, left_node=None,right_node=None, predicted_class=None):
        self.feature = feature              #takes feature index to split on
        self.threshold = threshold          #threshold value
        self.left_node = left_node          #for internal node while splitting, if the value of feature is less than threshold
        self.right_node = right_node
        self.predicted_class = predicted_class #only for leaf node(class labels)

#LOADING THE DATASET
iris_df = pd.read_csv("Iris.csv")
print(iris_df)
x = iris_df.drop(columns=['Id','Species'])
print(x.shape)
print(type(x))
y = iris_df[['Species']] #NO WAY THIS WAS THE BUG, y as a dataframe leads to improper splitting due to improper indexing
y = iris_df['Species']
print(type(y))
print(y.shape)


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
    best_information_gain = -inf

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
            ig = information_gain(y,left_y,right_y) #takes the labels as parameters calculates the parent entropy and child entropy to give ig
            print(f"feature:{feature},Threshold:{threshold},IG:{ig}\n")


            #choosing the best split
            if ig > best_information_gain:
                best_feature = feature
                best_threshold = threshold
                best_information_gain = ig
    print(f"Best split => feature : {best_feature}, Threshold: {best_threshold}, Information gain:{best_information_gain}")
    return best_feature , best_threshold , best_information_gain

#Recursively builts the Decision tree using the split logic and cls Node
def Build_tree(x,y,thresholds,depth=0, max_depth=None):

    #code block for stopping conditions
    if len(set(y)) == 1: #if all class labels in the node are same, create a leaf
        return Node(predicted_class=y.iloc[0])

    if max_depth is not None and depth >= max_depth: #if maximum depth exceeded
        return Node(predicted_class=y.mode()[0]) #returns the most common class in node


    #code block for best split
    best_feature, best_threshold, best_information_gain = split_logic(x,y,thresholds)

    if best_information_gain <= 0: #no best useful split is found, sets the leaf node values
        return Node(predicted_class=y.mode()[0])


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

def predict_class(Node, x):
     #if the current node is a leaf node, return the predicted class
     if Node.predicted_class is not None:
         return Node.predicted_class

     #if Node is an internal node, traverse the tree based on feature and threshold
     if x[Node.feature] <= Node.threshold:
         return predict_class(Node.left_node, x)
     else:
         return predict_class(Node.right_node, x)

#work on this code block
def calculate_accuracy(y_true, y_pred):
    # Ensure that y_true and y_pred are both lists or numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure both have the same length
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")

    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Return accuracy as the ratio of correct predictions to total predictions
    accuracy = correct_predictions / len(y_true)

    return accuracy

#train-test split model evaluation
def train_test_split_evaluation(x,y,test_size = 0.3, max_depth=None):
    #split the dataset into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=42)


    #Build the DT using the x_train
    thresholds = compute_thresholds(x_train)
    tree = Build_tree(x_train, y_train, thresholds, max_depth=max_depth)

    #make predicitions on the test data
    y_pred = [predict_class(tree,x_test.iloc[i]) for i in range(len(x_test))]


    #calculate accuracy
    accuracy = calculate_accuracy(y_test,y_pred)
    return accuracy

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

    accuracy = train_test_split_evaluation(x,y,test_size=0.3,max_depth=5)
    print(f"accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    main()




####improved code 2
import numpy as np
import pandas as pd
from math import inf
from sklearn.model_selection import train_test_split
from Lab10.Ex1 import information_gain


class Node:
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, predicted_class=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.predicted_class = predicted_class


# Load dataset
iris_df = pd.read_csv("Iris.csv")
x = iris_df.drop(columns=['Id', 'Species'])
y = iris_df['Species']


def compute_thresholds(x):
    thresholds = {}
    for feature in x.columns:
        unique_values = sorted(x[feature].unique())
        midpoints = [(unique_values[i] + unique_values[i + 1]) / 2.0 for i in range(len(unique_values) - 1)]
        thresholds[feature] = midpoints
    return thresholds


def split_logic(x, y, thresholds):
    best_feature = None
    best_threshold = None
    best_information_gain = -inf

    for feature in x.columns:
        for threshold in thresholds[feature]:
            left_split = x[feature] <= threshold
            right_split = x[feature] > threshold
            left_y, right_y = y[left_split], y[right_split]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            ig = information_gain(y, left_y, right_y)
            if ig > best_information_gain:
                best_feature, best_threshold, best_information_gain = feature, threshold, ig

    return best_feature, best_threshold, best_information_gain


def Build_tree(x, y, thresholds, depth=0, max_depth=None):
    if len(set(y)) == 1:
        return Node(predicted_class=y.values[0])
    if max_depth is not None and depth >= max_depth:
        return Node(predicted_class=y.mode()[0])

    best_feature, best_threshold, best_information_gain = split_logic(x, y, thresholds)

    if best_information_gain <= 0:
        return Node(predicted_class=y.mode()[0])

    left_split = x[best_feature] <= best_threshold
    right_split = x[best_feature] > best_threshold
    left_x, right_x = x[left_split], x[right_split]
    left_y, right_y = y[left_split], y[right_split]

    left_subtree = Build_tree(left_x, left_y, thresholds, depth + 1, max_depth)
    right_subtree = Build_tree(right_x, right_y, thresholds, depth + 1, max_depth)

    return Node(feature=best_feature, threshold=best_threshold, left_node=left_subtree, right_node=right_subtree)


def predict_class(node, x):
    if node.predicted_class is not None:
        return node.predicted_class
    if x[node.feature] <= node.threshold:
        return predict_class(node.left_node, x)
    else:
        return predict_class(node.right_node, x)


def calculate_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def train_test_split_evaluation(x, y, test_size=0.3, max_depth=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    thresholds = compute_thresholds(x_train)
    tree = Build_tree(x_train, y_train, thresholds, max_depth=max_depth)
    y_pred = [predict_class(tree, x_test.iloc[i]) for i in range(len(x_test))]
    return calculate_accuracy(y_test, y_pred)


def main():
    accuracy = train_test_split_evaluation(x, y, test_size=0.3, max_depth=5)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()

#
# ######Clean Code for both classification and regression setting
# import numpy as np
# import pandas as pd
# from math import inf
# from sklearn.datasets import load_iris, load_diabetes
# from sklearn.model_selection import train_test_split
#
#
# # Node class for decision tree
# class Node:
#     def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
#         self.feature = feature  # Feature index to split on
#         self.threshold = threshold  # Split threshold
#         self.left = left  # Left child node
#         self.right = right  # Right child node
#         self.value = value  # Predicted value (for leaves)
#
#
# # Function to compute unique midpoints for potential split thresholds
# def compute_thresholds(x):
#     thresholds = {}
#     for feature in x.columns:
#         unique_values = sorted(x[feature].unique())
#         midpoints = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]
#         thresholds[feature] = midpoints
#     return thresholds
#
#
# # Function to compute entropy for classification
# def entropy(y):
#     classes, counts = np.unique(y, return_counts=True)
#     probabilities = counts / len(y)
#     return -np.sum(probabilities * np.log2(probabilities))
#
#
# # Function to compute information gain for classification
# def information_gain(y, left_y, right_y):
#     parent_entropy = entropy(y)
#     left_entropy = entropy(left_y)
#     right_entropy = entropy(right_y)
#     weight_left = len(left_y) / len(y)
#     weight_right = len(right_y) / len(y)
#     return parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)
#
#
# # Function to find the best split for classification
# def best_split(x, y, thresholds):
#     best_feature, best_threshold, best_ig = None, None, -inf
#     for feature in x.columns:
#         for threshold in thresholds[feature]:
#             left_mask = x[feature] <= threshold
#             right_mask = x[feature] > threshold
#             left_y, right_y = y[left_mask], y[right_mask]
#             if len(left_y) == 0 or len(right_y) == 0:
#                 continue
#             ig = information_gain(y, left_y, right_y)
#             if ig > best_ig:
#                 best_feature, best_threshold, best_ig = feature, threshold, ig
#     return best_feature, best_threshold, best_ig
#
#
# # Function to build the decision tree recursively for classification
# def build_tree(x, y, thresholds, depth=0, max_depth=None):
#     if len(set(y)) == 1:
#         return Node(value=y.iloc[0])
#     if max_depth is not None and depth >= max_depth:
#         return Node(value=y.mode()[0])
#     best_feature, best_threshold, best_ig = best_split(x, y, thresholds)
#     if best_ig <= 0:
#         return Node(value=y.mode()[0])
#     left_mask = x[best_feature] <= best_threshold
#     right_mask = x[best_feature] > best_threshold
#     left_subtree = build_tree(x[left_mask], y[left_mask], thresholds, depth + 1, max_depth)
#     right_subtree = build_tree(x[right_mask], y[right_mask], thresholds, depth + 1, max_depth)
#     return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
#
#
# # Prediction function
# def predict(node, x):
#     if node.value is not None:
#         return node.value
#     if x[node.feature] <= node.threshold:
#         return predict(node.left, x)
#     else:
#         return predict(node.right, x)
#
#
# # Accuracy calculation
# def accuracy_score(y_true, y_pred):
#     return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
#
#
# # Train-test evaluation
# def train_test_eval(x, y, test_size=0.3, max_depth=None):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#     thresholds = compute_thresholds(x_train)
#     tree = build_tree(x_train, y_train, thresholds, max_depth=max_depth)
#     y_pred = [predict(tree, x_test.iloc[i]) for i in range(len(x_test))]
#     return accuracy_score(y_test, y_pred)
#
#
# # Main function for classification
# def classification_main():
#     iris = load_iris()
#     x = pd.DataFrame(iris.data, columns=iris.feature_names)
#     y = pd.Series(iris.target)
#     accuracy = train_test_eval(x, y, max_depth=5)
#     print(f"Decision Tree Classifier Accuracy: {accuracy:.2f}")
#
#
# # Main function for regression using MSE
# def regression_main():
#     diabetes = load_diabetes()
#     x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#     y = pd.Series(diabetes.target)
#
#     def mse(y):
#         return np.mean((y - y.mean()) ** 2)
#
#     def mse_reduction(y, left_y, right_y):
#         return mse(y) - (len(left_y) / len(y) * mse(left_y) + len(right_y) / len(y) * mse(right_y))
#
#     def best_split_regression(x, y, thresholds):
#         best_feature, best_threshold, best_mse_red = None, None, -inf
#         for feature in x.columns:
#             for threshold in thresholds[feature]:
#                 left_mask = x[feature] <= threshold
#                 right_mask = x[feature] > threshold
#                 left_y, right_y = y[left_mask], y[right_mask]
#                 if len(left_y) == 0 or len(right_y) == 0:
#                     continue
#                 mse_red = mse_reduction(y, left_y, right_y)
#                 if mse_red > best_mse_red:
#                     best_feature, best_threshold, best_mse_red = feature, threshold, mse_red
#         return best_feature, best_threshold, best_mse_red
#
#     def build_tree_regression(x, y, thresholds, depth=0, max_depth=None):
#         if max_depth is not None and depth >= max_depth:
#             return Node(value=y.mean())
#         best_feature, best_threshold, best_mse_red = best_split_regression(x, y, thresholds)
#         if best_mse_red <= 0:
#             return Node(value=y.mean())
#         left_mask = x[best_feature] <= best_threshold
#         right_mask = x[best_feature] > best_threshold
#         left_subtree = build_tree_regression(x[left_mask], y[left_mask], thresholds, depth + 1, max_depth)
#         right_subtree = build_tree_regression(x[right_mask], y[right_mask], thresholds, depth + 1, max_depth)
#         return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#     thresholds = compute_thresholds(x_train)
#     tree = build_tree_regression(x_train, y_train, thresholds, max_depth=5)
#     y_pred = [predict(tree, x_test.iloc[i]) for i in range(len(x_test))]
#     mse_error = mse(y_test - y_pred)
#     print(f"Decision Tree Regression MSE: {mse_error:.2f}")
#
#
# if __name__ == "__main__":
#     classification_main()
#     regression_main()
#
